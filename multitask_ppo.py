"""Multi-task PPO network architecture.

Implements a shared trunk encoder with task-specific policy and value heads.
The architecture enables transfer learning across manipulation tasks.
"""

from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks
from brax.training import distribution


@struct.dataclass
class MultiTaskPPONetworks:
    """Networks for multi-task PPO.
    
    Contains a shared trunk encoder that feeds into task-specific
    policy and value heads.
    """
    policy_network: nn.Module
    value_network: nn.Module
    parametric_action_distribution: distribution.ParametricDistribution


class SharedTrunkEncoder(nn.Module):
    """Shared encoder trunk for multi-task learning.
    
    Processes observations through shared layers to extract
    common features useful across all tasks.
    """
    layer_sizes: Sequence[int]
    activation: Callable[[jax.Array], jax.Array] = nn.swish
    
    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs
        for size in self.layer_sizes:
            x = nn.Dense(size, kernel_init=nn.initializers.he_normal())(x)
            x = nn.LayerNorm()(x)
            x = self.activation(x)
        return x


class TaskConditionedHead(nn.Module):
    """Task-conditioned policy or value head.
    
    Takes shared features and task encoding to produce task-specific outputs.
    Can optionally use learned task embeddings instead of one-hot.
    """
    output_size: int
    hidden_sizes: Sequence[int]
    num_tasks: int
    task_embedding_dim: int = 0  # 0 = use one-hot directly
    activation: Callable[[jax.Array], jax.Array] = nn.swish
    
    @nn.compact
    def __call__(
        self,
        shared_features: jax.Array,
        task_id: jax.Array,
    ) -> jax.Array:
        # Get task representation
        if self.task_embedding_dim > 0:
            # Learned task embedding
            task_embed = nn.Embed(
                num_embeddings=self.num_tasks,
                features=self.task_embedding_dim,
            )(task_id.astype(jnp.int32))
        else:
            # Use one-hot encoding directly
            task_embed = jax.nn.one_hot(task_id.astype(jnp.int32), self.num_tasks)
        
        # Concatenate shared features with task embedding
        x = jnp.concatenate([shared_features, task_embed], axis=-1)
        
        # Task-specific MLP
        for size in self.hidden_sizes:
            x = nn.Dense(size, kernel_init=nn.initializers.he_normal())(x)
            x = self.activation(x)
        
        # Output layer
        output = nn.Dense(
            self.output_size,
            kernel_init=nn.initializers.variance_scaling(
                0.01, "fan_in", "truncated_normal"
            ),
        )(x)
        
        return output


class MultiTaskPolicyNetwork(nn.Module):
    """Multi-task policy network with shared trunk.
    
    Architecture:
    1. Shared trunk processes full observation (including task one-hot)
    2. Task-conditioned head produces action distribution parameters
    """
    shared_layer_sizes: Sequence[int]
    head_layer_sizes: Sequence[int]
    action_size: int
    num_tasks: int
    task_embedding_dim: int = 0
    activation: Callable[[jax.Array], jax.Array] = nn.swish
    
    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        # Extract task ID from observation (last num_tasks dimensions are one-hot)
        task_onehot = obs[..., -self.num_tasks:]
        task_id = jnp.argmax(task_onehot, axis=-1)
        
        # Core observation (without task encoding, we'll re-add via conditioning)
        core_obs = obs[..., :-self.num_tasks]
        
        # Shared trunk encoder
        shared_features = SharedTrunkEncoder(
            layer_sizes=self.shared_layer_sizes,
            activation=self.activation,
        )(core_obs)
        
        # Task-conditioned policy head
        # Output: mean and log_std for each action dimension
        action_params = TaskConditionedHead(
            output_size=self.action_size * 2,  # mean + log_std
            hidden_sizes=self.head_layer_sizes,
            num_tasks=self.num_tasks,
            task_embedding_dim=self.task_embedding_dim,
            activation=self.activation,
        )(shared_features, task_id)
        
        # Split into mean and log_std, and clamp log_std to prevent numerical instability
        mean, log_std = jnp.split(action_params, 2, axis=-1)
        log_std = jnp.clip(log_std, -5.0, 2.0)
        action_params = jnp.concatenate([mean, log_std], axis=-1)
        
        return action_params


class MultiTaskValueNetwork(nn.Module):
    """Multi-task value network with shared trunk.
    
    Similar architecture to policy but outputs scalar value.
    """
    shared_layer_sizes: Sequence[int]
    head_layer_sizes: Sequence[int]
    num_tasks: int
    task_embedding_dim: int = 0
    activation: Callable[[jax.Array], jax.Array] = nn.swish
    
    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        task_onehot = obs[..., -self.num_tasks:]
        task_id = jnp.argmax(task_onehot, axis=-1)
        core_obs = obs[..., :-self.num_tasks]
        
        # Shared trunk
        shared_features = SharedTrunkEncoder(
            layer_sizes=self.shared_layer_sizes,
            activation=self.activation,
        )(core_obs)
        
        # Task-conditioned value head
        value = TaskConditionedHead(
            output_size=1,
            hidden_sizes=self.head_layer_sizes,
            num_tasks=self.num_tasks,
            task_embedding_dim=self.task_embedding_dim,
            activation=self.activation,
        )(shared_features, task_id)
        
        return jnp.squeeze(value, axis=-1)


def make_multitask_ppo_networks(
    observation_size: int,
    action_size: int,
    num_tasks: int,
    shared_layer_sizes: Sequence[int] = (256, 256),
    policy_head_sizes: Sequence[int] = (128,),
    value_head_sizes: Sequence[int] = (128,),
    task_embedding_dim: int = 0,
    activation: str = "swish",
) -> MultiTaskPPONetworks:
    """Factory function to create multi-task PPO networks.
    
    Args:
        observation_size: Size of observation vector (including task encoding)
        action_size: Size of action vector
        num_tasks: Number of tasks
        shared_layer_sizes: Hidden layer sizes for shared trunk
        policy_head_sizes: Hidden layer sizes for policy heads
        value_head_sizes: Hidden layer sizes for value heads
        task_embedding_dim: Dimension for learned task embeddings (0=one-hot)
        activation: Activation function name
        
    Returns:
        MultiTaskPPONetworks instance with policy and value networks
    """
    activation_fn = {
        "swish": nn.swish,
        "relu": nn.relu,
        "tanh": nn.tanh,
        "gelu": nn.gelu,
    }[activation]
    
    policy_network = MultiTaskPolicyNetwork(
        shared_layer_sizes=shared_layer_sizes,
        head_layer_sizes=policy_head_sizes,
        action_size=action_size,
        num_tasks=num_tasks,
        task_embedding_dim=task_embedding_dim,
        activation=activation_fn,
    )
    
    value_network = MultiTaskValueNetwork(
        shared_layer_sizes=shared_layer_sizes,
        head_layer_sizes=value_head_sizes,
        num_tasks=num_tasks,
        task_embedding_dim=task_embedding_dim,
        activation=activation_fn,
    )
    
    # Use tanh-squashed normal distribution for bounded actions
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    
    return MultiTaskPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


# Helper functions for Brax PPO integration
def make_inference_fn(networks: MultiTaskPPONetworks):
    """Create inference function for trained policy.
    
    Returns a function that takes observation and returns action.
    """
    def inference_fn(params, obs, rng):
        policy_params = params["policy"]
        action_dist_params = networks.policy_network.apply(policy_params, obs)
        action = networks.parametric_action_distribution.sample(
            action_dist_params, rng
        )
        return action
    
    return inference_fn


def make_policy_fn(networks: MultiTaskPPONetworks):
    """Create policy function returning action distribution.
    
    Used during training for computing log probabilities.
    """
    def policy_fn(params, obs):
        policy_params = params["policy"]
        action_dist_params = networks.policy_network.apply(policy_params, obs)
        return action_dist_params
    
    return policy_fn


def make_value_fn(networks: MultiTaskPPONetworks):
    """Create value function.
    
    Used during training for computing advantages.
    """
    def value_fn(params, obs):
        value_params = params["value"]
        return networks.value_network.apply(value_params, obs)
    
    return value_fn

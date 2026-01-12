"""Baseline single-task PPO training script.

Trains separate policies for each task to compare against multi-task learning.
Uses the same network architecture (minus task conditioning) for fair comparison.
"""

import functools
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import flax
from flax import linen as nn
import optax
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
import ml_collections

from envs.tasks import get_task_env
from brax.training import distribution


@flax.struct.dataclass
class SingleTaskTrainingState:
    """State for single-task PPO training."""
    params: Dict[str, Any]
    opt_state: optax.OptState
    env_state: Any
    rng: jax.Array
    step: int
    obs_mean: jax.Array
    obs_var: jax.Array
    obs_count: jax.Array


@flax.struct.dataclass
class Transition:
    """Single transition for PPO."""
    obs: jax.Array  # Normalized observations for policy/value
    obs_raw: jax.Array  # Raw observations for statistics update
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    log_prob: jax.Array


class SingleTaskPolicyNetwork(nn.Module):
    """Standard MLP policy network for single task."""
    layer_sizes: tuple
    action_size: int
    
    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs
        for size in self.layer_sizes:
            x = nn.Dense(size)(x)
            x = nn.LayerNorm()(x)
            x = nn.swish(x)
        
        # Output mean and log_std
        output = nn.Dense(self.action_size * 2)(x)
        return output


class SingleTaskValueNetwork(nn.Module):
    """Standard MLP value network for single task."""
    layer_sizes: tuple
    
    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = obs
        for size in self.layer_sizes:
            x = nn.Dense(size)(x)
            x = nn.LayerNorm()(x)
            x = nn.swish(x)
        
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, axis=-1)


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation."""
    def scan_fn(carry, inputs):
        next_value, gae = carry
        reward, value, done = inputs
        
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        
        return (value, gae), gae
    
    _, advantages = lax.scan(
        scan_fn,
        (next_value, jnp.zeros_like(next_value)),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    
    return advantages, returns


def train_single_task(
    task_name: str,
    config: ml_collections.ConfigDict,
    log_dir: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Train PPO on a single task.
    
    Args:
        task_name: Name of the task to train on
        config: Training configuration
        log_dir: Directory for logs
        
    Returns:
        params: Trained parameters
        metrics_history: Training metrics
    """
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"\n{'='*60}")
    print(f"Training baseline for task: {task_name}")
    print(f"{'='*60}")
    
    # Initialize RNG
    rng = jax.random.PRNGKey(config.training.seed)
    rng, rng_env, rng_net = jax.random.split(rng, 3)
    
    # Create single-task environment
    env = get_task_env(task_name)
    
    print(f"Observation size: {env.observation_size}")
    print(f"Action size: {env.action_size}")
    
    # Network sizes (same total capacity as multi-task but no task conditioning)
    # Use same total hidden units: shared (256,256) + head (128) ≈ (256, 256, 128)
    layer_sizes = tuple(config.network.shared_layer_sizes) + tuple(config.network.policy_head_sizes)
    
    # Create networks
    policy_network = SingleTaskPolicyNetwork(
        layer_sizes=layer_sizes,
        action_size=env.action_size,
    )
    value_network = SingleTaskValueNetwork(
        layer_sizes=layer_sizes,
    )
    
    action_dist = distribution.NormalTanhDistribution(event_size=env.action_size)
    
    # Initialize parameters
    dummy_obs = jnp.zeros((1, env.observation_size))
    policy_params = policy_network.init(rng_net, dummy_obs)
    rng_net, rng_val = jax.random.split(rng_net)
    value_params = value_network.init(rng_val, dummy_obs)
    
    params = {"policy": policy_params, "value": value_params}
    
    # Optimizer
    learning_rate = optax.linear_schedule(
        init_value=config.ppo.learning_rate,
        end_value=config.ppo.learning_rate * 0.1,
        transition_steps=config.training.num_timesteps // config.ppo.unroll_length,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.ppo.max_grad_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(params)
    
    # Vectorize environment
    num_envs = config.training.num_envs
    reset_fn = jax.vmap(env.reset)
    step_fn = jax.vmap(env.step)
    
    # Initialize environments
    rng_envs = jax.random.split(rng_env, num_envs)
    env_state = reset_fn(rng_envs)
    
    # Initialize training state
    training_state = SingleTaskTrainingState(
        params=params,
        opt_state=opt_state,
        env_state=env_state,
        rng=rng,
        step=0,
        obs_mean=jnp.zeros(env.observation_size),
        obs_var=jnp.ones(env.observation_size),
        obs_count=jnp.array(1e-4),
    )
    
    # PPO loss function
    def ppo_loss(params, batch, advantages, returns):
        policy_out = policy_network.apply(params["policy"], batch.obs)
        log_prob = action_dist.log_prob(policy_out, batch.action)
        entropy = action_dist.entropy(policy_out, jax.random.PRNGKey(0))
        
        ratio = jnp.exp(log_prob - batch.log_prob)
        adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        pg_loss1 = -adv_normalized * ratio
        pg_loss2 = -adv_normalized * jnp.clip(ratio, 1 - config.ppo.clip_epsilon, 1 + config.ppo.clip_epsilon)
        policy_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        
        values = value_network.apply(params["value"], batch.obs)
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        total_loss = policy_loss + config.ppo.value_loss_coef * value_loss - config.ppo.entropy_cost * entropy.mean()
        
        return total_loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy.mean(),
        }
    
    # JIT compile training step
    @functools.partial(jax.jit, donate_argnums=(0,))
    def training_step(state: SingleTaskTrainingState):
        rng, rng_rollout = jax.random.split(state.rng)
        
        # Collect rollout
        def rollout_step(carry, _):
            env_state, rng = carry
            rng, rng_action = jax.random.split(rng)
            
            obs = env_state.obs
            obs_normalized = (obs - state.obs_mean) / (jnp.sqrt(state.obs_var) + 1e-8)
            
            policy_out = policy_network.apply(state.params["policy"], obs_normalized)
            action = action_dist.sample(policy_out, rng_action)
            log_prob = action_dist.log_prob(policy_out, action)
            value = value_network.apply(state.params["value"], obs_normalized)
            
            next_env_state = step_fn(env_state, action)
            
            transition = Transition(
                obs=obs_normalized,
                obs_raw=obs,
                action=action,
                reward=next_env_state.reward * config.ppo.reward_scaling,
                done=next_env_state.done,
                value=value,
                log_prob=log_prob,
            )
            
            # Reset done environments
            rng, rng_reset = jax.random.split(rng)
            reset_rngs = jax.random.split(rng_reset, num_envs)
            reset_states = reset_fn(reset_rngs)
            
            def select_reset_or_current(reset_val, current_val):
                # Reshape done to broadcast with arrays of any shape
                done = next_env_state.done
                # Add extra dimensions to match the array shape
                for _ in range(reset_val.ndim - 1):
                    done = done[..., None]
                return jnp.where(done, reset_val, current_val)
            
            new_env_state = jax.tree_util.tree_map(
                select_reset_or_current,
                reset_states,
                next_env_state,
            )
            
            return (new_env_state, rng), transition
        
        (new_env_state, _), transitions = lax.scan(
            rollout_step,
            (state.env_state, rng_rollout),
            None,
            length=config.ppo.unroll_length,
        )
        
        # Compute GAE
        final_obs = new_env_state.obs
        final_obs_normalized = (final_obs - state.obs_mean) / (jnp.sqrt(state.obs_var) + 1e-8)
        bootstrap_value = value_network.apply(state.params["value"], final_obs_normalized)
        
        advantages, returns = compute_gae(
            transitions.reward,
            transitions.value,
            transitions.done,
            bootstrap_value,
            gamma=config.ppo.discounting,
            gae_lambda=config.ppo.gae_lambda,
        )
        
        # Flatten batch
        batch_size = config.ppo.unroll_length * num_envs
        flat_transitions = jax.tree_util.tree_map(
            lambda x: x.reshape(batch_size, *x.shape[2:]),
            transitions,
        )
        flat_advantages = advantages.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)
        
        # PPO update
        def update_step(carry, _):
            params, opt_state, rng = carry
            rng, rng_perm = jax.random.split(rng)
            perm = jax.random.permutation(rng_perm, batch_size)
            
            def minibatch_update(carry, indices):
                params, opt_state = carry
                
                mb_transitions = jax.tree_util.tree_map(lambda x: x[indices], flat_transitions)
                mb_advantages = flat_advantages[indices]
                mb_returns = flat_returns[indices]
                
                (loss, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                    params, mb_transitions, mb_advantages, mb_returns
                )
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                
                return (new_params, new_opt_state), (loss, metrics)
            
            minibatch_size = batch_size // config.ppo.num_minibatches
            minibatch_indices = perm.reshape(config.ppo.num_minibatches, minibatch_size)
            
            (params, opt_state), (losses, metrics) = lax.scan(
                minibatch_update, (params, opt_state), minibatch_indices
            )
            
            return (params, opt_state, rng), (losses, metrics)
        
        rng, rng_update = jax.random.split(rng)
        (new_params, new_opt_state, _), (all_losses, all_metrics) = lax.scan(
            update_step,
            (state.params, state.opt_state, rng_update),
            None,
            length=config.ppo.num_updates_per_batch,
        )
        
        # Update observation statistics using RAW observations (not normalized)
        new_obs_batch = transitions.obs_raw.reshape(-1, env.observation_size)
        batch_mean = new_obs_batch.mean(axis=0)
        batch_var = new_obs_batch.var(axis=0)
        batch_count = new_obs_batch.shape[0]
        
        delta = batch_mean - state.obs_mean
        total_count = state.obs_count + batch_count
        new_mean = state.obs_mean + delta * batch_count / total_count
        m_a = state.obs_var * state.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * state.obs_count * batch_count / total_count
        new_var = M2 / total_count
        
        metrics = {
            "loss/total": all_losses.mean(),
            "loss/policy": all_metrics["policy_loss"].mean(),
            "loss/value": all_metrics["value_loss"].mean(),
            "policy/entropy": all_metrics["entropy"].mean(),
            "reward/mean": transitions.reward.mean(),
            "reward/max": transitions.reward.max(),
            "success": transitions.done.mean(),  # Approximate success rate
        }
        
        new_state = SingleTaskTrainingState(
            params=new_params,
            opt_state=new_opt_state,
            env_state=new_env_state,
            rng=rng,
            step=state.step + 1,
            obs_mean=new_mean,
            obs_var=new_var,
            obs_count=total_count,
        )
        
        return new_state, metrics
    
    # Training loop
    num_iterations = config.training.num_timesteps // (config.ppo.unroll_length * num_envs)
    metrics_history = []
    
    print(f"Training for {num_iterations} iterations...")
    start_time = time.time()
    
    for i in range(num_iterations):
        training_state, metrics = training_step(training_state)
        
        if i % config.training.log_frequency == 0:
            elapsed = time.time() - start_time
            timesteps = (i + 1) * config.ppo.unroll_length * num_envs
            fps = timesteps / elapsed
            
            metrics_np = {k: float(v) for k, v in metrics.items()}
            metrics_history.append({"step": timesteps, **metrics_np})
            
            for key, value in metrics_np.items():
                writer.add_scalar(key, value, timesteps)
            
            print(
                f"[{task_name}] Step {timesteps:>8} | "
                f"Reward: {metrics_np['reward/mean']:>8.3f} | "
                f"FPS: {fps:>6.0f}"
            )
    
    total_time = time.time() - start_time
    print(f"[{task_name}] Training complete in {total_time / 60:.1f} minutes")
    
    writer.close()
    
    final_params = {
        "params": training_state.params,
        "obs_mean": training_state.obs_mean,
        "obs_var": training_state.obs_var,
    }
    
    return final_params, metrics_history


def train_baseline(
    config: ml_collections.ConfigDict,
    log_dir: Optional[str] = None,
) -> Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Train separate baseline policies for all tasks.
    
    Args:
        config: Training configuration (should use baseline config)
        log_dir: Base directory for logs
        
    Returns:
        results: Dict mapping task name to (params, metrics_history)
    """
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(
            config.logging.log_dir,
            f"{config.logging.experiment_name}_{timestamp}",
        )
    
    results = {}
    
    for task_name in config.env.tasks:
        task_log_dir = os.path.join(log_dir, task_name)
        params, metrics = train_single_task(task_name, config, task_log_dir)
        results[task_name] = (params, metrics)
        
        # Save checkpoint
        checkpoint_dir = os.path.join(task_log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(os.path.join(checkpoint_dir, "final"), params)
    
    print(f"\n{'='*60}")
    print("All baseline training complete!")
    print(f"{'='*60}")
    
    # Summary
    for task_name, (_, metrics) in results.items():
        final_reward = metrics[-1]["reward/mean"] if metrics else 0
        print(f"  {task_name}: Final reward = {final_reward:.3f}")
    
    return results


if __name__ == "__main__":
    from config import get_baseline_config
    
    config = get_baseline_config()
    train_baseline(config)

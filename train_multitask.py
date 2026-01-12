"""Multi-task PPO training script.

Trains a single policy with shared representation across multiple
manipulation tasks using PPO with vectorized environments on GPU.
"""

import functools
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import flax
from flax import linen as nn
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
import ml_collections

from envs.multitask_wrapper import MultiTaskEnv, TaskBatch, create_multitask_env
from multitask_ppo import (
    MultiTaskPPONetworks,
    make_multitask_ppo_networks,
    make_inference_fn,
)


@flax.struct.dataclass
class Transition:
    """Single transition for PPO."""
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    log_prob: jax.Array


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Rewards [T, B]
        values: Value estimates [T, B]
        dones: Done flags [T, B]
        next_value: Bootstrap value [B]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: GAE advantages [T, B]
        returns: Value targets [T, B]
    """
    def scan_fn(carry, inputs):
        next_value, gae = carry
        reward, value, done = inputs
        
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        
        return (value, gae), gae
    
    # Scan backwards through time
    _, advantages = lax.scan(
        scan_fn,
        (next_value, jnp.zeros_like(next_value)),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    
    return advantages, returns


def ppo_loss(
    params: Dict[str, Any],
    networks: MultiTaskPPONetworks,
    batch: Transition,
    advantages: jax.Array,
    returns: jax.Array,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Compute PPO loss."""
    # Policy forward pass
    policy_params = networks.policy_network.apply(params["policy"], batch.obs)
    
    # Get new log probs and entropy
    log_prob = networks.parametric_action_distribution.log_prob(
        policy_params, batch.action
    )
    entropy = networks.parametric_action_distribution.entropy(
        policy_params, jax.random.PRNGKey(0)
    )
    
    # Policy loss with clipping
    # CRITICAL: Clip log_ratio to small range to prevent exploding ratios
    log_ratio = jnp.clip(log_prob - batch.log_prob, -5.0, 5.0)
    ratio = jnp.exp(log_ratio)
    
    # Normalize advantages
    advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    pg_loss1 = -advantages_normalized * ratio
    pg_loss2 = -advantages_normalized * jnp.clip(
        ratio, 1 - clip_epsilon, 1 + clip_epsilon
    )
    policy_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
    
    # Value loss with clipping to prevent explosion
    values = networks.value_network.apply(params["value"], batch.obs)
    value_pred_clipped = batch.value + jnp.clip(
        values - batch.value, -clip_epsilon, clip_epsilon
    )
    value_loss1 = (values - returns) ** 2
    value_loss2 = (value_pred_clipped - returns) ** 2
    value_loss = 0.5 * jnp.maximum(value_loss1, value_loss2).mean()
    
    # Entropy bonus (ensure it's not negative)
    entropy_loss = -jnp.maximum(entropy.mean(), 0.0)
    
    # Total loss
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    
    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy.mean(),
        "approx_kl": ((ratio - 1) - jnp.log(jnp.clip(ratio, 1e-8, 1e8))).mean(),
        "clip_fraction": (jnp.abs(ratio - 1) > clip_epsilon).mean(),
    }
    
    return total_loss, metrics


def train_multitask(
    config: ml_collections.ConfigDict,
    log_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train multi-task PPO policy.
    
    Args:
        config: Training configuration
        log_dir: Directory for logs and checkpoints
        
    Returns:
        params: Trained network parameters
        metrics: Training metrics history
    """
    # Setup logging
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(
            config.logging.log_dir,
            f"{config.logging.experiment_name}_{timestamp}",
        )
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    print(f"Logging to: {log_dir}")
    print(f"Config: {config}")
    
    # Initialize RNG
    rng = jax.random.PRNGKey(config.training.seed)
    rng, rng_env, rng_net = jax.random.split(rng, 3)
    
    # Create multi-task environment
    num_envs = config.training.num_envs
    env = create_multitask_env(
        task_names=config.env.tasks,
        num_envs=num_envs,
    )
    
    print(f"Tasks: {env.task_names}")
    print(f"Observation size: {env.observation_size}")
    print(f"Action size: {env.action_size}")
    print(f"Num envs: {env.num_envs}")
    
    # Create networks
    networks = make_multitask_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        num_tasks=env.num_tasks,
        shared_layer_sizes=config.network.shared_layer_sizes,
        policy_head_sizes=config.network.policy_head_sizes,
        value_head_sizes=config.network.value_head_sizes,
        task_embedding_dim=config.network.task_embedding_dim,
        activation=config.network.activation,
    )
    
    # Initialize parameters
    dummy_obs = jnp.zeros((1, env.observation_size))
    policy_params = networks.policy_network.init(rng_net, dummy_obs)
    rng_net, rng_val = jax.random.split(rng_net)
    value_params = networks.value_network.init(rng_val, dummy_obs)
    
    params = {
        "policy": policy_params,
        "value": value_params,
    }
    
    # Count parameters
    def count_params(p):
        return sum(x.size for x in jax.tree_util.tree_leaves(p))
    
    total_params = count_params(params)
    print(f"Total parameters: {total_params:,}")
    
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
    
    # Initialize environments
    print("Initializing environments...")
    task_batches = env.reset(rng_env)
    print("Environments initialized!")
    
    # Observation normalization stats
    obs_mean = jnp.zeros(env.observation_size)
    obs_var = jnp.ones(env.observation_size)
    obs_count = 1e-4
    
    # JIT compile policy and value functions
    @jax.jit
    def get_action_and_value(params, obs, rng):
        """Get action, log_prob, and value for a batch of observations."""
        obs_batch = obs  # [num_envs, obs_dim]
        
        policy_params = networks.policy_network.apply(params["policy"], obs_batch)
        
        # DON'T use nan_to_num on policy_params - it destroys log_std structure
        # The network already clips log_std in [-5, 2]
        
        action = networks.parametric_action_distribution.sample(policy_params, rng)
        log_prob = networks.parametric_action_distribution.log_prob(policy_params, action)
        value = networks.value_network.apply(params["value"], obs_batch)
        
        # Clip actions to valid range (tanh already bounds to [-1,1] but be safe)
        action = jnp.clip(action, -1.0, 1.0)
        
        return action, log_prob, value
    
    @jax.jit
    def get_value(params, obs):
        """Get value for a batch of observations."""
        return networks.value_network.apply(params["value"], obs)
    
    @jax.jit
    def ppo_update(params, opt_state, transitions, advantages, returns, rng):
        """Single PPO update step."""
        batch_size = transitions.obs.shape[0]
        
        def update_epoch(carry, _):
            params, opt_state, rng = carry
            rng, rng_perm = jax.random.split(rng)
            
            perm = jax.random.permutation(rng_perm, batch_size)
            
            def minibatch_update(carry, indices):
                params, opt_state = carry
                
                mb_transitions = jax.tree_util.tree_map(
                    lambda x: x[indices], transitions
                )
                mb_advantages = advantages[indices]
                mb_returns = returns[indices]
                
                loss_fn = functools.partial(
                    ppo_loss,
                    networks=networks,
                    batch=mb_transitions,
                    advantages=mb_advantages,
                    returns=mb_returns,
                    clip_epsilon=config.ppo.clip_epsilon,
                    entropy_coef=config.ppo.entropy_cost,
                    value_coef=config.ppo.value_loss_coef,
                )
                
                (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                
                return (new_params, new_opt_state), (loss, metrics)
            
            minibatch_size = batch_size // config.ppo.num_minibatches
            minibatch_indices = perm.reshape(config.ppo.num_minibatches, minibatch_size)
            
            (params, opt_state), (losses, metrics) = lax.scan(
                minibatch_update,
                (params, opt_state),
                minibatch_indices,
            )
            
            return (params, opt_state, rng), (losses, metrics)
        
        (new_params, new_opt_state, _), (all_losses, all_metrics) = lax.scan(
            update_epoch,
            (params, opt_state, rng),
            None,
            length=config.ppo.num_updates_per_batch,
        )
        
        metrics = {
            "loss/total": all_losses.mean(),
            "loss/policy": all_metrics["policy_loss"].mean(),
            "loss/value": all_metrics["value_loss"].mean(),
            "policy/entropy": all_metrics["entropy"].mean(),
            "policy/approx_kl": all_metrics["approx_kl"].mean(),
            "policy/clip_fraction": all_metrics["clip_fraction"].mean(),
        }
        
        return new_params, new_opt_state, metrics
    
    # Training loop
    num_iterations = config.training.num_timesteps // (config.ppo.unroll_length * num_envs)
    metrics_history = []
    
    print(f"\nStarting training for {num_iterations} iterations...")
    print(f"  - {num_envs} parallel environments")
    print(f"  - {config.ppo.unroll_length} steps per rollout")
    print(f"  - {config.ppo.unroll_length * num_envs} timesteps per iteration")
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Collect rollout
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_log_probs = []
        
        for step in range(config.ppo.unroll_length):
            # Get combined observations from all tasks
            obs = env.get_obs(task_batches)
            
            # Normalize observations (with NaN protection)
            obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            obs_normalized = (obs - obs_mean) / (jnp.sqrt(obs_var) + 1e-8)
            obs_normalized = jnp.clip(obs_normalized, -10.0, 10.0)  # Clip extreme values
            
            # Get actions and values
            rng, rng_action = jax.random.split(rng)
            actions, log_probs, values = get_action_and_value(params, obs_normalized, rng_action)
            
            # Protect against NaN in actions
            actions = jnp.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
            actions = jnp.clip(actions, -1.0, 1.0)
            
            # Store transition data
            all_obs.append(obs_normalized)
            all_actions.append(actions)
            all_values.append(values)
            all_log_probs.append(log_probs)
            
            # Step environments
            task_batches = env.step(task_batches, actions)
            
            # Get rewards and dones after step
            rewards = env.get_rewards(task_batches)
            dones = env.get_dones(task_batches)
            
            # Protect against NaN rewards
            rewards = jnp.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
            
            all_rewards.append(rewards * config.ppo.reward_scaling)
            all_dones.append(dones)
            
            # Reset done environments
            rng, rng_reset = jax.random.split(rng)
            task_batches = env.reset_done(task_batches, rng_reset)
        
        # Stack rollout data: [T, B, ...]
        rollout_obs = jnp.stack(all_obs)  # [T, B, obs_dim]
        rollout_actions = jnp.stack(all_actions)  # [T, B, action_dim]
        rollout_rewards = jnp.stack(all_rewards)  # [T, B]
        rollout_dones = jnp.stack(all_dones)  # [T, B]
        rollout_values = jnp.stack(all_values)  # [T, B]
        rollout_log_probs = jnp.stack(all_log_probs)  # [T, B]
        
        # Bootstrap value
        final_obs = env.get_obs(task_batches)
        final_obs_normalized = (final_obs - obs_mean) / (jnp.sqrt(obs_var) + 1e-8)
        bootstrap_value = get_value(params, final_obs_normalized)
        
        # Compute GAE
        advantages, returns = compute_gae(
            rollout_rewards,
            rollout_values,
            rollout_dones,
            bootstrap_value,
            gamma=config.ppo.discounting,
            gae_lambda=config.ppo.gae_lambda,
        )
        
        # Flatten for PPO update
        batch_size = config.ppo.unroll_length * num_envs
        flat_transitions = Transition(
            obs=rollout_obs.reshape(batch_size, -1),
            action=rollout_actions.reshape(batch_size, -1),
            reward=rollout_rewards.reshape(batch_size),
            done=rollout_dones.reshape(batch_size),
            value=rollout_values.reshape(batch_size),
            log_prob=rollout_log_probs.reshape(batch_size),
        )
        flat_advantages = advantages.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)
        
        # PPO update
        rng, rng_update = jax.random.split(rng)
        params, opt_state, update_metrics = ppo_update(
            params, opt_state, flat_transitions, flat_advantages, flat_returns, rng_update
        )
        
        # Update observation statistics
        new_obs_batch = rollout_obs.reshape(-1, env.observation_size)
        batch_mean = new_obs_batch.mean(axis=0)
        batch_var = new_obs_batch.var(axis=0)
        batch_count = new_obs_batch.shape[0]
        
        delta = batch_mean - obs_mean
        total_count = obs_count + batch_count
        obs_mean = obs_mean + delta * batch_count / total_count
        m_a = obs_var * obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * obs_count * batch_count / total_count
        obs_var = M2 / total_count
        obs_count = total_count
        
        # Compute reward metrics
        mean_reward = float(rollout_rewards.mean())
        
        # Per-task rewards
        task_rewards = {}
        task_ids = env.get_task_ids()
        for i, task_name in enumerate(env.task_names):
            # Find envs for this task
            start_idx = sum(env._envs_per_task[:i])
            end_idx = start_idx + env._envs_per_task[i]
            task_reward = float(rollout_rewards[:, start_idx:end_idx].mean())
            task_rewards[f"reward/{task_name}"] = task_reward
        
        # Combine metrics
        metrics = {
            **{k: float(v) for k, v in update_metrics.items()},
            "reward/mean": mean_reward,
            **task_rewards,
        }
        
        # Log progress
        iter_time = time.time() - iter_start
        timesteps = (iteration + 1) * config.ppo.unroll_length * num_envs
        elapsed = time.time() - start_time
        fps = timesteps / elapsed
        
        print(
            f"[iter {iteration:>4}/{num_iterations}] "
            f"Reward: {mean_reward:>7.3f} | "
            f"Policy Loss: {metrics['loss/policy']:>7.4f} | "
            f"Value Loss: {metrics['loss/value']:>7.4f} | "
            f"Entropy: {metrics['policy/entropy']:>5.3f} | "
            f"FPS: {fps:>6.0f}"
        )
        
        # TensorBoard logging
        if iteration % config.training.log_frequency == 0:
            metrics_history.append({"step": timesteps, **metrics})
            for key, value in metrics.items():
                writer.add_scalar(key, value, timesteps)
            writer.add_scalar("perf/fps", fps, timesteps)
        
        # Checkpoint
        if iteration % max(1, config.training.checkpoint_frequency // (config.ppo.unroll_length * num_envs)) == 0:
            checkpoint_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpointer = ocp.StandardCheckpointer()
            save_path = os.path.join(checkpoint_dir, f"step_{timesteps}")
            checkpointer.save(
                save_path,
                {"params": params, "obs_mean": obs_mean, "obs_var": obs_var},
            )
            print(f"  Saved checkpoint to {save_path}")
    
    # Final results
    final_params = {"params": params, "obs_mean": obs_mean, "obs_var": obs_var}
    
    total_time = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Final reward: {mean_reward:.3f}")
    
    writer.close()
    
    return final_params, metrics_history


if __name__ == "__main__":
    from config import get_config
    config = get_config()
    train_multitask(config)

"""Multi-task environment wrapper for vectorized training.

This module provides JAX-compatible multi-task environments.

Key insight: Different MuJoCo models have different state shapes (nq, nv, etc.),
and JAX can't trace through Python control flow. We use a simpler approach:

1. Each task batch is managed separately with pre-compiled reset/step functions
2. The MultiTaskEnv provides helper methods that work OUTSIDE of jax.jit
3. The training loop orchestrates the calls to each task's functions

This allows the training loop to handle the batching externally while still
using JIT-compiled per-task operations.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import MjxEnv, State

from envs.tasks import get_task_env


class TaskBatch(NamedTuple):
    """State for a single task's batch of environments."""
    state: State  # Batched state from vmap
    obs: jax.Array  # [batch_size, obs_dim] - augmented observations
    reward: jax.Array  # [batch_size]
    done: jax.Array  # [batch_size]


def _augment_obs_batched(
    obs: jax.Array,
    task_id: int,
    num_tasks: int,
    max_obs_size: int,
) -> jax.Array:
    """Pad observation and add one-hot task encoding (batched version)."""
    batch_size = obs.shape[0]
    obs_size = obs.shape[-1]
    
    # Replace NaN/Inf with zeros for safety
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Pad observations
    obs_padded = jnp.zeros((batch_size, max_obs_size))
    obs_padded = obs_padded.at[:, :obs_size].set(obs)
    
    # Add one-hot task encoding
    task_onehot = jnp.tile(jax.nn.one_hot(task_id, num_tasks), (batch_size, 1))
    
    return jnp.concatenate([obs_padded, task_onehot], axis=1)


class MultiTaskEnv:
    """Multi-task environment with balanced task distribution.
    
    This class manages multiple task environments and provides JIT-compiled
    reset and step functions for each task. The training loop should:
    
    1. Call reset() to initialize all environments
    2. Use get_obs(), get_rewards(), get_dones() to get combined arrays
    3. Call step() with actions to advance all environments
    4. Call reset_done() to reset completed episodes
    
    Example:
        env = MultiTaskEnv(task_names=['reach', 'push'], num_envs=512)
        state = env.reset(rng)
        
        # In training loop:
        obs = env.get_obs(state)
        actions = policy(obs)
        state = env.step(state, actions)
        state = env.reset_done(state, rng)
    """
    
    def __init__(
        self,
        task_names: Sequence[str],
        num_envs: int = 512,
        task_weights: Optional[Sequence[float]] = None,
        reward_scales: Optional[Sequence[float]] = None,
        **env_kwargs,
    ):
        self._task_names = list(task_names)
        self._num_tasks = len(task_names)
        self._num_envs = num_envs
        
        if reward_scales is None:
            reward_scales = [1.0] * self._num_tasks
        self._reward_scales = list(reward_scales)
        
        # Calculate envs per task (balanced distribution)
        base_envs_per_task = num_envs // self._num_tasks
        extra = num_envs % self._num_tasks
        
        self._envs_per_task = []
        for i in range(self._num_tasks):
            count = base_envs_per_task + (1 if i < extra else 0)
            self._envs_per_task.append(count)
        
        # Create base environments and compute sizes
        self._base_envs = [get_task_env(name, **env_kwargs) for name in task_names]
        
        self._obs_sizes = []
        self._action_sizes = []
        for env in self._base_envs:
            obs_size = env.observation_size
            if isinstance(obs_size, dict):
                obs_size = sum(v[-1] if isinstance(v, tuple) else v for v in obs_size.values())
            self._obs_sizes.append(obs_size)
            self._action_sizes.append(env.action_size)
        
        self._max_obs_size = max(self._obs_sizes)
        self._max_action_size = max(self._action_sizes)
        
        # Pre-compile JIT functions for each task
        self._reset_fns = []
        self._step_fns = []
        
        for task_id, env in enumerate(self._base_envs):
            # Create reset function
            def make_reset_fn(env):
                @jax.jit
                def reset_fn(rng):
                    return env.reset(rng)
                return reset_fn
            
            # Create step function
            def make_step_fn(env, action_size):
                @jax.jit
                def step_fn(state, action):
                    task_action = action[:action_size]
                    return env.step(state, task_action)
                return step_fn
            
            self._reset_fns.append(jax.vmap(make_reset_fn(env)))
            self._step_fns.append(jax.vmap(make_step_fn(env, self._action_sizes[task_id])))
        
        # Task assignment for each env index
        self._task_assignment = []
        for task_id, count in enumerate(self._envs_per_task):
            self._task_assignment.extend([task_id] * count)
        self._task_assignment = jnp.array(self._task_assignment)
        
        # Compute slice indices for each task
        self._task_slices = []
        offset = 0
        for count in self._envs_per_task:
            self._task_slices.append((offset, offset + count))
            offset += count
    
    @property
    def num_tasks(self) -> int:
        return self._num_tasks
    
    @property
    def task_names(self) -> List[str]:
        return self._task_names
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def observation_size(self) -> int:
        return self._max_obs_size + self._num_tasks
    
    @property
    def action_size(self) -> int:
        return self._max_action_size
    
    def reset(self, rng: jax.Array) -> List[TaskBatch]:
        """Reset all environments.
        
        Returns:
            List of TaskBatch, one per task
        """
        rngs = jax.random.split(rng, self._num_envs)
        
        task_batches = []
        rng_offset = 0
        
        for task_id, count in enumerate(self._envs_per_task):
            if count > 0:
                task_rngs = rngs[rng_offset:rng_offset + count]
                state = self._reset_fns[task_id](task_rngs)
                
                # Augment observations
                obs = _augment_obs_batched(
                    state.obs, task_id, self._num_tasks, self._max_obs_size
                )
                
                task_batches.append(TaskBatch(
                    state=state,
                    obs=obs,
                    reward=state.reward * self._reward_scales[task_id],
                    done=state.done,
                ))
            else:
                task_batches.append(None)
            rng_offset += count
        
        return task_batches
    
    def step(self, task_batches: List[TaskBatch], actions: jax.Array) -> List[TaskBatch]:
        """Step all environments.
        
        Args:
            task_batches: List of TaskBatch from reset() or previous step()
            actions: Actions array [num_envs, action_size]
            
        Returns:
            Updated list of TaskBatch
        """
        new_batches = []
        action_offset = 0
        
        for task_id, count in enumerate(self._envs_per_task):
            if count > 0:
                task_actions = actions[action_offset:action_offset + count]
                old_batch = task_batches[task_id]
                
                # Step environment
                new_state = self._step_fns[task_id](old_batch.state, task_actions)
                
                # Augment observations
                obs = _augment_obs_batched(
                    new_state.obs, task_id, self._num_tasks, self._max_obs_size
                )
                
                new_batches.append(TaskBatch(
                    state=new_state,
                    obs=obs,
                    reward=new_state.reward * self._reward_scales[task_id],
                    done=new_state.done,
                ))
            else:
                new_batches.append(None)
            action_offset += count
        
        return new_batches
    
    def reset_done(self, task_batches: List[TaskBatch], rng: jax.Array) -> List[TaskBatch]:
        """Reset environments that are done.
        
        Args:
            task_batches: List of TaskBatch
            rng: Random key for resets
            
        Returns:
            Updated list of TaskBatch with done envs reset
        """
        rngs = jax.random.split(rng, self._num_envs)
        new_batches = []
        rng_offset = 0
        
        for task_id, count in enumerate(self._envs_per_task):
            if count > 0:
                task_rngs = rngs[rng_offset:rng_offset + count]
                old_batch = task_batches[task_id]
                done_mask = old_batch.done
                
                # Reset all (we'll select based on done mask)
                reset_state = self._reset_fns[task_id](task_rngs)
                
                # Merge based on done mask - handle arbitrary shapes
                def select(reset_val, old_val):
                    if reset_val.ndim == 0:
                        return reset_val  # Scalar, just use reset
                    elif reset_val.ndim == 1:
                        return jnp.where(done_mask, reset_val, old_val)
                    else:
                        # Reshape done_mask to broadcast with multi-dimensional arrays
                        # done_mask is [batch], we need to expand it to match reset_val shape
                        expanded_mask = done_mask.reshape(
                            (count,) + (1,) * (reset_val.ndim - 1)
                        )
                        return jnp.where(expanded_mask, reset_val, old_val)
                
                merged_state = jax.tree_util.tree_map(
                    select, reset_state, old_batch.state
                )
                
                # Augment observations
                obs = _augment_obs_batched(
                    merged_state.obs, task_id, self._num_tasks, self._max_obs_size
                )
                
                new_batches.append(TaskBatch(
                    state=merged_state,
                    obs=obs,
                    reward=merged_state.reward * self._reward_scales[task_id],
                    done=merged_state.done,
                ))
            else:
                new_batches.append(None)
            rng_offset += count
        
        return new_batches
    
    def get_obs(self, task_batches: List[TaskBatch]) -> jax.Array:
        """Get combined observations from all tasks."""
        obs_list = [b.obs for b in task_batches if b is not None]
        return jnp.concatenate(obs_list, axis=0)
    
    def get_rewards(self, task_batches: List[TaskBatch]) -> jax.Array:
        """Get combined rewards from all tasks."""
        reward_list = [b.reward for b in task_batches if b is not None]
        return jnp.concatenate(reward_list, axis=0)
    
    def get_dones(self, task_batches: List[TaskBatch]) -> jax.Array:
        """Get combined done flags from all tasks."""
        done_list = [b.done for b in task_batches if b is not None]
        return jnp.concatenate(done_list, axis=0)
    
    def get_task_ids(self) -> jax.Array:
        """Get task ID for each env."""
        return self._task_assignment


def create_multitask_env(
    task_names: Sequence[str] = ("reach", "push", "pick_place", "peg_insert"),
    num_envs: int = 512,
    **kwargs,
) -> MultiTaskEnv:
    """Factory function for multi-task environment."""
    return MultiTaskEnv(task_names=task_names, num_envs=num_envs, **kwargs)

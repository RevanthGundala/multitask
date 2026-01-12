"""Reach task: Move end-effector to target position using Panda robot.

A simple manipulation task where the Panda robot arm must reach a randomly
placed target position in 3D space.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src.mjx_env import State

from envs.tasks import register_task
from envs.tasks.panda_base import PandaBaseEnv, PANDA_HOME_QPOS


def default_config() -> config_dict.ConfigDict:
    """Returns the default config for reach task."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,  # Match mujoco_playground (200Hz)
        episode_length=100,  # Shorter episodes - reset before drifting too far
        action_repeat=1,
        action_scale=0.04,  # Delta control scale - small for smooth motion
        reward_config=config_dict.create(
            scales=config_dict.create(
                distance=4.0,  # Weight for reaching target
                robot_target_qpos=1.0,  # Stronger: keep robot near home pose
                success_bonus=10.0,
            )
        ),
    )


class ReachEnv(PandaBaseEnv):
    """Reach task environment using Panda robot.
    
    The robot must move its end-effector to reach a randomly placed target.
    Reward is based on negative distance to target with bonus for being close.
    """
    
    # Task-specific objects: just a target sphere
    _TASK_OBJECTS_XML = """
    <!-- Target (visual only, randomized position) -->
    <body name="target" pos="0.5 0.0 0.5" mocap="true">
      <geom name="target_geom" type="sphere" size="0.03" material="target" contype="0" conaffinity="0"/>
      <site name="target_site" pos="0 0 0" size="0.02"/>
    </body>
    """
    
    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_config()
        super().__init__(config, config_overrides)
        
        # Target-specific indices
        self._target_mocap_idx = self._mj_model.body("target").mocapid[0]
        self._target_site_idx = self._mj_model.site("target_site").id
        
        # Target sampling range - CLOSER to home position for easier learning
        # Home EE position is approximately [0.3, 0, 0.5]
        # Start with small range, can increase later as curriculum
        self._target_min = jp.array([0.2, -0.15, 0.35])
        self._target_max = jp.array([0.5, 0.15, 0.65])
        
    @property
    def observation_size(self) -> int:
        # arm qpos (7) + arm qvel (7) + gripper (1) + ee_pos (3) + ee_to_target (3) = 21
        return 21
    
    def reset(self, rng: jax.Array) -> State:
        """Reset environment to initial state with random target."""
        rng, rng_target, rng_arm = jax.random.split(rng, 3)
        
        # Random target position in reachable workspace
        target_pos = jax.random.uniform(
            rng_target,
            shape=(3,),
            minval=self._target_min,
            maxval=self._target_max,
        )
        
        # Initialize robot at home position with gripper open
        data = self._create_initial_data(rng_arm, arm_qpos=PANDA_HOME_QPOS, gripper_open=0.04)
        
        # Set target mocap body position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._target_mocap_idx].set(target_pos)
        )
        
        # Forward kinematics
        data = mjx.forward(self._mjx_model, data)
        
        # Build observation
        obs = self._get_obs(data, target_pos)
        
        # Initialize metrics
        distance = self._compute_distance(data, target_pos)
        metrics = {
            "distance": distance,
            "success": jp.float32(0),
        }
        info = {
            "rng": rng,
            "target_pos": target_pos,
            "step": jp.int32(0),
        }
        
        return State(
            data=data,
            obs=obs,
            reward=jp.float32(0),
            done=jp.float32(0),
            metrics=metrics,
            info=info,
        )
    
    def step(self, state: State, action: jax.Array) -> State:
        """Take environment step."""
        target_pos = state.info["target_pos"]
        
        # Apply action and step physics
        data = self._apply_action(state, action)
        
        # Check physics validity
        physics_valid = self._check_physics_valid(data)
        
        # Compute observation and reward
        obs = self._get_obs(data, target_pos)
        distance = self._compute_distance(data, target_pos)
        
        # Handle physics explosion
        distance = jp.where(physics_valid, distance, 2.0)
        
        # Reward shaping using tanh (same as mujoco_playground)
        # This gives values in [0, 1] with strong gradients near goal
        scales = self._config.reward_config.scales
        reach_reward = 1.0 - jp.tanh(5.0 * distance)
        reward = scales.distance * reach_reward
        
        # Keep robot near home pose (prevents drifting from random actions)
        arm_qpos = self._get_arm_qpos(data)
        qpos_error = jp.linalg.norm(arm_qpos - PANDA_HOME_QPOS)
        robot_target_qpos = 1.0 - jp.tanh(qpos_error)
        reward += scales.robot_target_qpos * robot_target_qpos
        
        # Success if within 3cm of target
        success = jp.float32(distance < 0.03)
        reward += scales.success_bonus * success
        
        # Penalize physics explosion
        reward = jp.where(physics_valid, reward, -10.0)
        
        # Episode termination
        step = state.info["step"] + 1
        done = jp.float32((step >= self._config.episode_length) | ~physics_valid)
        
        metrics = {
            "distance": distance,
            "success": success,
        }
        info = {
            "rng": state.info["rng"],
            "target_pos": target_pos,
            "step": step,
        }
        
        return State(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )
    
    def _get_obs(self, data: mjx.Data, target_pos: jax.Array) -> jax.Array:
        """Construct observation vector."""
        arm_qpos = self._get_arm_qpos(data)
        arm_qvel = self._get_arm_qvel(data)
        gripper_pos = self._get_gripper_pos(data)
        ee_pos = self._get_ee_pos(data)
        
        # Relative position from EE to target (direction to move)
        ee_to_target = target_pos - ee_pos
        
        return jp.concatenate([
            arm_qpos,           # 7
            arm_qvel,           # 7
            jp.array([gripper_pos]),  # 1
            ee_pos,             # 3
            ee_to_target,       # 3 (more useful than absolute target_pos)
        ])
    
    def _compute_distance(self, data: mjx.Data, target_pos: jax.Array) -> jax.Array:
        """Compute distance from end-effector to target."""
        ee_pos = self._get_ee_pos(data)
        return jp.linalg.norm(ee_pos - target_pos)


@register_task("reach")
def create_reach_env(**kwargs) -> ReachEnv:
    """Factory function for reach environment."""
    config = kwargs.pop("config", None)
    config_overrides = kwargs.pop("config_overrides", None)
    return ReachEnv(config=config, config_overrides=config_overrides)

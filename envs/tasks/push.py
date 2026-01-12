"""Push task: Push an object to a goal location using Panda robot.

The Panda robot arm must push a cube to a randomly placed goal position on the table.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State

from envs.tasks import register_task
from envs.tasks.panda_base import PandaBaseEnv, PANDA_HOME_QPOS


def default_config() -> config_dict.ConfigDict:
    """Returns the default config for push task."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,  # Match mujoco_playground (200Hz)
        episode_length=300,
        action_repeat=1,
        action_scale=0.04,  # Delta control scale - small for smooth motion
        reward_config=config_dict.create(
            scales=config_dict.create(
                cube_to_goal=4.0,
                ee_to_cube=1.0,
                success_bonus=10.0,
            )
        ),
    )


class PushEnv(PandaBaseEnv):
    """Push task environment using Panda robot.
    
    The robot must push a cube to a goal location on the table surface.
    """
    
    # Task-specific objects: pushable cube and goal marker
    _TASK_OBJECTS_XML = """
    <!-- Pushable cube -->
    <body name="cube" pos="0.5 0 0.025">
      <joint name="cube_joint" type="free"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1" material="object"
            friction="0.5 0.005 0.0001" condim="3"/>
      <site name="cube_site" pos="0 0 0" size="0.01"/>
    </body>
    
    <!-- Goal marker (visual only) -->
    <body name="goal" pos="0.6 0.2 0.001" mocap="true">
      <geom name="goal_geom" type="cylinder" size="0.04 0.002" material="goal" contype="0" conaffinity="0"/>
      <site name="goal_site" pos="0 0 0" size="0.02"/>
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
        
        # Indices
        self._cube_body_idx = self._mj_model.body("cube").id
        self._goal_mocap_idx = self._mj_model.body("goal").mocapid[0]
        self._cube_qposadr = self._mj_model.jnt_qposadr[self._mj_model.joint("cube_joint").id]
        
        # Sampling ranges (in robot's reachable workspace)
        self._cube_range = jp.array([[0.4, -0.2], [0.6, 0.2]])
        self._goal_range = jp.array([[0.4, -0.3], [0.7, 0.3]])
    
    @property
    def observation_size(self) -> int:
        # arm qpos (7) + arm qvel (7) + gripper (1) + ee_pos (3) + cube_pos (3) + goal_pos (2) = 23
        return 23
    
    def reset(self, rng: jax.Array) -> State:
        """Reset environment with random cube and goal positions."""
        rng, rng_cube, rng_goal, rng_arm = jax.random.split(rng, 4)
        
        # Random cube position on table
        cube_xy = jax.random.uniform(
            rng_cube, (2,), minval=self._cube_range[0], maxval=self._cube_range[1]
        )
        cube_pos = jp.array([cube_xy[0], cube_xy[1], 0.025])
        
        # Random goal position (different from cube)
        goal_xy = jax.random.uniform(
            rng_goal, (2,), minval=self._goal_range[0], maxval=self._goal_range[1]
        )
        goal_pos = jp.array([goal_xy[0], goal_xy[1], 0.001])
        
        # Initialize robot
        data = self._create_initial_data(rng_arm, arm_qpos=PANDA_HOME_QPOS, gripper_open=0.0)
        
        # Set cube position (free joint: pos + quat)
        data = data.replace(
            qpos=data.qpos.at[self._cube_qposadr:self._cube_qposadr+3].set(cube_pos)
        )
        data = data.replace(
            qpos=data.qpos.at[self._cube_qposadr+3:self._cube_qposadr+7].set(jp.array([1, 0, 0, 0]))
        )
        
        # Set goal mocap position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._goal_mocap_idx].set(goal_pos)
        )
        
        data = mjx.forward(self._mjx_model, data)
        
        obs = self._get_obs(data, goal_pos)
        
        metrics = {
            "cube_to_goal": self._cube_goal_distance(data, goal_pos),
            "success": jp.float32(0),
        }
        info = {
            "rng": rng,
            "goal_pos": goal_pos,
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
        goal_pos = state.info["goal_pos"]
        
        # Apply action
        data = self._apply_action(state, action)
        
        # Check physics validity
        physics_valid = self._check_physics_valid(data)
        
        obs = self._get_obs(data, goal_pos)
        
        # Distances
        cube_to_goal = self._cube_goal_distance(data, goal_pos)
        ee_to_cube = self._ee_cube_distance(data)
        
        # Handle physics explosion
        cube_to_goal = jp.where(physics_valid, cube_to_goal, 2.0)
        ee_to_cube = jp.where(physics_valid, ee_to_cube, 2.0)
        
        # Shaped reward
        scales = self._config.reward_config.scales
        reward = -scales.cube_to_goal * cube_to_goal
        reward += -scales.ee_to_cube * ee_to_cube
        
        # Success bonus
        success = jp.float32(cube_to_goal < 0.05)
        reward += scales.success_bonus * success
        
        # Penalize physics explosion
        reward = jp.where(physics_valid, reward, -10.0)
        
        step = state.info["step"] + 1
        done = jp.float32((step >= self._config.episode_length) | ~physics_valid)
        
        # Replace NaN observations
        obs = jp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return State(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics={"cube_to_goal": cube_to_goal, "success": success},
            info={"rng": state.info["rng"], "goal_pos": goal_pos, "step": step},
        )
    
    def _get_obs(self, data: mjx.Data, goal_pos: jax.Array) -> jax.Array:
        """Construct observation."""
        arm_qpos = self._get_arm_qpos(data)
        arm_qvel = self._get_arm_qvel(data)
        gripper_pos = self._get_gripper_pos(data)
        ee_pos = self._get_ee_pos(data)
        cube_pos = data.xpos[self._cube_body_idx]
        
        return jp.concatenate([
            arm_qpos,           # 7
            arm_qvel,           # 7
            jp.array([gripper_pos]),  # 1
            ee_pos,             # 3
            cube_pos,           # 3
            goal_pos[:2],       # 2
        ])
    
    def _cube_goal_distance(self, data: mjx.Data, goal_pos: jax.Array) -> jax.Array:
        """2D distance from cube to goal."""
        cube_pos = data.xpos[self._cube_body_idx]
        return jp.linalg.norm(cube_pos[:2] - goal_pos[:2])
    
    def _ee_cube_distance(self, data: mjx.Data) -> jax.Array:
        """Distance from end-effector to cube."""
        ee_pos = self._get_ee_pos(data)
        cube_pos = data.xpos[self._cube_body_idx]
        return jp.linalg.norm(ee_pos - cube_pos)


@register_task("push")
def create_push_env(**kwargs) -> PushEnv:
    """Factory function for push environment."""
    config = kwargs.pop("config", None)
    config_overrides = kwargs.pop("config_overrides", None)
    return PushEnv(config=config, config_overrides=config_overrides)

"""Pick-and-place task: Grasp an object and place it at a goal location using Panda robot.

The Panda robot must pick up a cube and place it at a raised goal position.
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
    """Returns the default config for pick-place task."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=400,
        action_repeat=1,
        action_scale=1.0,
        reward_config=config_dict.create(
            scales=config_dict.create(
                cube_to_goal=4.0,
                ee_to_cube=2.0,
                grasp_bonus=5.0,
                success_bonus=10.0,
            )
        ),
    )


class PickPlaceEnv(PandaBaseEnv):
    """Pick-and-place task environment using Panda robot.
    
    The robot must grasp a cube and place it at an elevated goal position.
    """
    
    # Task-specific objects: graspable cube and elevated goal
    _TASK_OBJECTS_XML = """
    <!-- Graspable cube -->
    <body name="cube" pos="0.5 0 0.025">
      <joint name="cube_joint" type="free"/>
      <geom name="cube_geom" type="box" size="0.02 0.02 0.02" mass="0.05" material="object"
            friction="1.0 0.005 0.0001" condim="4"/>
      <site name="cube_site" pos="0 0 0" size="0.01"/>
    </body>
    
    <!-- Goal position (elevated) -->
    <body name="goal" pos="0.5 0.3 0.15" mocap="true">
      <geom name="goal_geom" type="sphere" size="0.03" material="goal" contype="0" conaffinity="0"/>
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
        
        # Sampling ranges
        self._cube_range = jp.array([[0.4, -0.2], [0.6, 0.2]])
        self._goal_range = jp.array([[0.35, 0.1, 0.1], [0.65, 0.35, 0.25]])
    
    @property
    def observation_size(self) -> int:
        # arm qpos (7) + arm qvel (7) + gripper (1) + ee_pos (3) + cube_pos (3) + goal_pos (3) = 24
        return 24
    
    def reset(self, rng: jax.Array) -> State:
        """Reset with random cube and goal positions."""
        rng, rng_cube, rng_goal, rng_arm = jax.random.split(rng, 4)
        
        # Random cube on table
        cube_xy = jax.random.uniform(
            rng_cube, (2,), minval=self._cube_range[0], maxval=self._cube_range[1]
        )
        cube_pos = jp.array([cube_xy[0], cube_xy[1], 0.02])
        
        # Random elevated goal
        goal_pos = jax.random.uniform(
            rng_goal, (3,), minval=self._goal_range[0], maxval=self._goal_range[1]
        )
        
        # Initialize robot with gripper open
        data = self._create_initial_data(rng_arm, arm_qpos=PANDA_HOME_QPOS, gripper_open=0.04)
        
        # Set cube position
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
            "cube_to_goal": jp.linalg.norm(cube_pos - goal_pos),
            "grasped": jp.float32(0),
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
        cube_pos = data.xpos[self._cube_body_idx]
        ee_pos = self._get_ee_pos(data)
        cube_to_goal = jp.linalg.norm(cube_pos - goal_pos)
        ee_to_cube = jp.linalg.norm(ee_pos - cube_pos)
        
        # Handle physics explosion
        cube_to_goal = jp.where(physics_valid, cube_to_goal, 2.0)
        ee_to_cube = jp.where(physics_valid, ee_to_cube, 2.0)
        
        # Check if grasped (cube lifted off table)
        grasped = jp.float32(cube_pos[2] > 0.05) * jp.float32(physics_valid)
        
        # Reward
        scales = self._config.reward_config.scales
        reward = -scales.cube_to_goal * cube_to_goal
        reward += -scales.ee_to_cube * ee_to_cube
        reward += scales.grasp_bonus * grasped
        
        # Success: cube near goal
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
            metrics={"cube_to_goal": cube_to_goal, "grasped": grasped, "success": success},
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
            goal_pos,           # 3
        ])


@register_task("pick_place")
def create_pick_place_env(**kwargs) -> PickPlaceEnv:
    """Factory function for pick-place environment."""
    config = kwargs.pop("config", None)
    config_overrides = kwargs.pop("config_overrides", None)
    return PickPlaceEnv(config=config, config_overrides=config_overrides)

"""Peg insertion task: Insert a peg into a hole using Panda robot.

A precision manipulation task where the Panda robot must insert a held peg
into a tight hole.
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
    """Returns the default config for peg insertion task."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=400,
        action_repeat=1,
        action_scale=1.0,
        reward_config=config_dict.create(
            scales=config_dict.create(
                peg_to_hole=4.0,
                alignment=2.0,
                insertion_depth=8.0,
                success_bonus=10.0,
            )
        ),
    )


class PegInsertEnv(PandaBaseEnv):
    """Peg insertion task environment using Panda robot.
    
    The robot holds a peg and must align and insert it into a hole.
    """
    
    # Task-specific objects: peg attached to gripper and socket with hole
    _TASK_OBJECTS_XML = """
    <!-- Peg (held by gripper - attached to end effector) -->
    <body name="peg" pos="0.5 0 0.3">
      <joint name="peg_joint" type="free"/>
      <geom name="peg_geom" type="cylinder" size="0.008 0.05" mass="0.02" 
            rgba="0.3 0.6 0.8 1" friction="0.5 0.005 0.0001" condim="4"/>
      <site name="peg_tip" pos="0 0 -0.05" size="0.005"/>
      <site name="peg_base" pos="0 0 0.05" size="0.005"/>
    </body>
    
    <!-- Socket with hole (on table) -->
    <body name="socket" pos="0.5 0.2 0.025" mocap="true">
      <geom name="socket_base" type="cylinder" size="0.05 0.025" rgba="0.5 0.5 0.5 1" 
            contype="0" conaffinity="0"/>
      <!-- Hole visualization - the actual hole is just the socket_center site -->
      <geom name="hole_visual" type="cylinder" pos="0 0 0.026" size="0.012 0.001" 
            rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0"/>
      <site name="hole_center" pos="0 0 0.025" size="0.01"/>
      <site name="hole_bottom" pos="0 0 -0.02" size="0.005"/>
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
        self._peg_tip_idx = self._mj_model.site("peg_tip").id
        self._peg_base_idx = self._mj_model.site("peg_base").id
        self._hole_center_idx = self._mj_model.site("hole_center").id
        self._hole_bottom_idx = self._mj_model.site("hole_bottom").id
        self._socket_mocap_idx = self._mj_model.body("socket").mocapid[0]
        self._peg_body_idx = self._mj_model.body("peg").id
        self._peg_qposadr = self._mj_model.jnt_qposadr[self._mj_model.joint("peg_joint").id]
        
        # Socket position range (in robot's reachable workspace)
        self._socket_range = jp.array([[0.4, -0.2], [0.6, 0.2]])
    
    @property
    def observation_size(self) -> int:
        # arm qpos (7) + arm qvel (7) + gripper (1) + peg_tip (3) + hole_center (3) = 21
        return 21
    
    def reset(self, rng: jax.Array) -> State:
        """Reset with random socket position and peg in gripper."""
        rng, rng_socket, rng_arm = jax.random.split(rng, 3)
        
        # Random socket position on table
        socket_xy = jax.random.uniform(
            rng_socket, (2,), minval=self._socket_range[0], maxval=self._socket_range[1]
        )
        socket_pos = jp.array([socket_xy[0], socket_xy[1], 0.025])
        
        # Initialize robot with gripper closed (holding peg)
        data = self._create_initial_data(rng_arm, arm_qpos=PANDA_HOME_QPOS, gripper_open=0.008)
        
        # Position peg near the end effector (it will be "held")
        ee_pos = data.site_xpos[self._end_effector_idx]
        peg_pos = ee_pos + jp.array([0, 0, -0.06])  # Below end effector
        
        # Set peg position and orientation (vertical)
        data = data.replace(
            qpos=data.qpos.at[self._peg_qposadr:self._peg_qposadr+3].set(peg_pos)
        )
        data = data.replace(
            qpos=data.qpos.at[self._peg_qposadr+3:self._peg_qposadr+7].set(jp.array([1, 0, 0, 0]))
        )
        
        # Set socket mocap position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._socket_mocap_idx].set(socket_pos)
        )
        
        data = mjx.forward(self._mjx_model, data)
        
        obs = self._get_obs(data)
        
        peg_tip = data.site_xpos[self._peg_tip_idx]
        hole_center = data.site_xpos[self._hole_center_idx]
        
        metrics = {
            "peg_to_hole": jp.linalg.norm(peg_tip[:2] - hole_center[:2]),
            "insertion_depth": jp.float32(0),
            "success": jp.float32(0),
        }
        info = {
            "rng": rng,
            "socket_pos": socket_pos,
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
        # Apply action
        data = self._apply_action(state, action)
        
        # Check physics validity
        physics_valid = self._check_physics_valid(data)
        
        obs = self._get_obs(data)
        
        # Positions
        peg_tip = data.site_xpos[self._peg_tip_idx]
        hole_center = data.site_xpos[self._hole_center_idx]
        hole_bottom = data.site_xpos[self._hole_bottom_idx]
        
        # Distances
        peg_to_hole_xy = jp.linalg.norm(peg_tip[:2] - hole_center[:2])
        
        # Handle physics explosion
        peg_to_hole_xy = jp.where(physics_valid, peg_to_hole_xy, 2.0)
        
        # Insertion depth (how far peg has descended below hole surface)
        insertion_depth = jp.maximum(0.0, hole_center[2] - peg_tip[2])
        max_depth = hole_center[2] - hole_bottom[2]
        normalized_depth = insertion_depth / (max_depth + 1e-6)
        normalized_depth = jp.where(physics_valid, normalized_depth, 0.0)
        normalized_depth = jp.clip(normalized_depth, 0.0, 1.0)
        
        # Alignment reward (peg should be above hole)
        alignment = jp.exp(-10 * peg_to_hole_xy)
        
        # Reward
        scales = self._config.reward_config.scales
        reward = -scales.peg_to_hole * peg_to_hole_xy
        reward += scales.alignment * alignment
        reward += scales.insertion_depth * normalized_depth
        
        # Success: peg inserted >80% depth and aligned
        success = jp.float32((normalized_depth > 0.8) & (peg_to_hole_xy < 0.02))
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
            metrics={
                "peg_to_hole": peg_to_hole_xy,
                "insertion_depth": normalized_depth,
                "success": success,
            },
            info={
                "rng": state.info["rng"],
                "socket_pos": state.info["socket_pos"],
                "step": step,
            },
        )
    
    def _get_obs(self, data: mjx.Data) -> jax.Array:
        """Construct observation."""
        arm_qpos = self._get_arm_qpos(data)
        arm_qvel = self._get_arm_qvel(data)
        gripper_pos = self._get_gripper_pos(data)
        peg_tip = data.site_xpos[self._peg_tip_idx]
        hole_center = data.site_xpos[self._hole_center_idx]
        
        return jp.concatenate([
            arm_qpos,           # 7
            arm_qvel,           # 7
            jp.array([gripper_pos]),  # 1
            peg_tip,            # 3
            hole_center,        # 3
        ])


@register_task("peg_insert")
def create_peg_insert_env(**kwargs) -> PegInsertEnv:
    """Factory function for peg insertion environment."""
    config = kwargs.pop("config", None)
    config_overrides = kwargs.pop("config_overrides", None)
    return PegInsertEnv(config=config, config_overrides=config_overrides)

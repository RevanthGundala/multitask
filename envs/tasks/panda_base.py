"""Base Panda robot environment using MuJoCo Menagerie.

Provides a shared Franka Emika Panda robot model for all manipulation tasks.
Loads the MJX-compatible Panda model directly from MuJoCo Menagerie,
using the same asset-loading approach as mujoco_playground.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State


# Menagerie Panda directory name
_MENAGERIE_FRANKA_DIR = "franka_emika_panda"


def get_panda_assets() -> Dict[str, bytes]:
    """Load all Panda robot assets from MuJoCo Menagerie.
    
    Uses the same approach as mujoco_playground to load assets as bytes.
    This allows MuJoCo to resolve all includes and mesh files properly.
    
    Returns:
        Dictionary mapping filenames to their binary contents.
    """
    # Ensure menagerie exists (auto-downloads if needed)
    mjx_env.ensure_menagerie_exists()
    
    assets = {}
    
    # Load all XML files from menagerie franka directory
    panda_path = mjx_env.MENAGERIE_PATH / _MENAGERIE_FRANKA_DIR
    mjx_env.update_assets(assets, panda_path, "*.xml")
    
    # Load mesh assets from the assets subfolder
    assets_path = panda_path / "assets"
    if assets_path.exists():
        mjx_env.update_assets(assets, assets_path)
    
    return assets


def get_panda_model_with_objects(task_xml: str = "") -> tuple[mujoco.MjModel, Dict[str, bytes]]:
    """Get Panda MjModel with additional task-specific objects.
    
    Creates an XML that includes the Menagerie Panda model and adds
    task-specific objects to the worldbody. Uses the mujoco_playground
    asset-loading approach for proper mesh resolution.
    
    Args:
        task_xml: XML string defining task-specific bodies/geoms to add
        
    Returns:
        Tuple of (MjModel, assets dict) for the task scene
    """
    # Load assets (this auto-downloads menagerie if needed)
    assets = get_panda_assets()
    
    # Build scene XML that includes the Menagerie mjx_panda.xml
    # The assets dict allows MuJoCo to resolve all file references
    scene_xml = f"""
<mujoco model="panda_task_scene">
  <include file="mjx_panda.xml"/>

  <statistic center="0.3 0 0.4" extent="1"/>

  <option timestep="0.005" iterations="5" ls_iterations="8" integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>
  
  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <rgba force="1 0 0 1"/>
    <global azimuth="-40" elevation="-30"/>
  </visual>
  
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
    <material name="target" rgba="0.2 0.8 0.2 0.8"/>
    <material name="object" rgba="0.8 0.2 0.2 1"/>
    <material name="goal" rgba="0.2 0.8 0.2 0.5"/>
  </asset>
  
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1"/>
    
    <!-- Task-specific objects -->
    {task_xml}
  </worldbody>
</mujoco>
"""
    
    # Load model using assets dict (resolves all includes and meshes)
    mj_model = mujoco.MjModel.from_xml_string(scene_xml, assets=assets)
    return mj_model, assets


# Constants for Panda robot
PANDA_NUM_JOINTS = 7  # 7 arm joints
PANDA_NUM_FINGERS = 2  # 2 finger joints (coupled)
PANDA_NUM_ACTUATORS = 9  # 7 arm + 2 finger actuators
PANDA_ACTION_SIZE = 8  # 7 arm + 1 gripper (fingers are coupled)

# Default home position for Panda (arm joints only)
PANDA_HOME_QPOS = jp.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Joint limits for Panda
PANDA_JOINT_LIMITS = jp.array([
    [-2.8973, 2.8973],   # joint1
    [-1.7628, 1.7628],   # joint2
    [-2.8973, 2.8973],   # joint3
    [-3.0718, -0.0698],  # joint4
    [-2.8973, 2.8973],   # joint5
    [-0.0175, 3.7525],   # joint6
    [-2.8973, 2.8973],   # joint7
])


def default_panda_config() -> config_dict.ConfigDict:
    """Returns the default config for Panda-based environments."""
    return config_dict.create(
        ctrl_dt=0.02,  # 50Hz control
        sim_dt=0.005,  # 200Hz simulation (matches mujoco_playground)
        episode_length=200,
        action_repeat=1,
        action_scale=0.04,  # Small delta for smooth motion
    )


class PandaBaseEnv(mjx_env.MjxEnv):
    """Base class for Panda manipulation tasks.
    
    Provides common functionality for loading the Panda robot model
    and handling observations/actions. Loads the Panda model directly
    from MuJoCo Menagerie.
    """
    
    # Subclasses should override this with task-specific objects
    _TASK_OBJECTS_XML = ""
    
    def __init__(
        self,
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        if config is None:
            config = default_panda_config()
        super().__init__(config, config_overrides)
        
        # Load Panda model from MuJoCo Menagerie with task objects
        # Uses asset-loading approach from mujoco_playground
        mj_model, self._model_assets = get_panda_model_with_objects(self._TASK_OBJECTS_XML)
        mj_model.opt.timestep = self.sim_dt
        
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)
        
        # Cache commonly used indices - Menagerie mjx_panda uses "gripper" site
        # Try different possible names for end-effector site
        for site_name in ["gripper", "pinch", "attachment_site", "end_effector"]:
            try:
                self._end_effector_idx = mj_model.site(site_name).id
                break
            except KeyError:
                continue
        else:
            raise RuntimeError("Could not find end-effector site in model")
        
        # Grasp center - use same as end effector for simplicity
        self._grasp_center_idx = self._end_effector_idx
        
        # Find joint indices - Menagerie mjx_panda uses "joint1" etc
        self._arm_joint_ids = []
        for i in range(1, 8):
            for prefix in ["", "panda_"]:
                try:
                    jid = mj_model.joint(f"{prefix}joint{i}").id
                    self._arm_joint_ids.append(jid)
                    break
                except KeyError:
                    continue
        
        if len(self._arm_joint_ids) != 7:
            raise RuntimeError(f"Expected 7 arm joints, found {len(self._arm_joint_ids)}")
        
        # Finger joints - Menagerie uses "finger_joint1" and "finger_joint2"
        for prefix in ["", "panda_"]:
            try:
                self._finger_left_id = mj_model.joint(f"{prefix}finger_joint1").id
                self._finger_right_id = mj_model.joint(f"{prefix}finger_joint2").id
                break
            except KeyError:
                continue
        
        # Get qpos addresses for arm and fingers
        self._arm_qposadr = [mj_model.jnt_qposadr[jid] for jid in self._arm_joint_ids]
        self._finger_left_qposadr = mj_model.jnt_qposadr[self._finger_left_id]
        self._finger_right_qposadr = mj_model.jnt_qposadr[self._finger_right_id]
        
        # Cache actuator control limits for delta control
        self._ctrl_lowers, self._ctrl_uppers = mj_model.actuator_ctrlrange.T
        
        # Store action scale from config
        self._action_scale = config.action_scale
        
    @property
    def xml_path(self) -> str:
        return "inline:panda_scene"
    
    @property
    def action_size(self) -> int:
        # 7 arm joints + 1 gripper (fingers coupled)
        return PANDA_ACTION_SIZE
    
    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model
    
    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
    
    def _get_arm_qpos(self, data: mjx.Data) -> jax.Array:
        """Get arm joint positions (7 values)."""
        return jp.array([data.qpos[adr] for adr in self._arm_qposadr])
    
    def _get_arm_qvel(self, data: mjx.Data) -> jax.Array:
        """Get arm joint velocities (7 values)."""
        # qvel addresses are same as dof ids for hinge joints
        return jp.array([data.qvel[adr] for adr in self._arm_qposadr])
    
    def _get_gripper_pos(self, data: mjx.Data) -> jax.Array:
        """Get gripper opening (single value, 0=closed, 0.04=open)."""
        return data.qpos[self._finger_left_qposadr]
    
    def _get_ee_pos(self, data: mjx.Data) -> jax.Array:
        """Get end-effector position in world frame."""
        return data.site_xpos[self._end_effector_idx]
    
    def _get_ee_quat(self, data: mjx.Data) -> jax.Array:
        """Get end-effector orientation as quaternion."""
        # site_xmat is 3x3 rotation matrix, convert to quat
        xmat = data.site_xmat[self._end_effector_idx].reshape(3, 3)
        return self._mat_to_quat(xmat)
    
    def _mat_to_quat(self, mat: jax.Array) -> jax.Array:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
        
        def case_pos_trace():
            s = jp.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (mat[2, 1] - mat[1, 2]) / s
            y = (mat[0, 2] - mat[2, 0]) / s
            z = (mat[1, 0] - mat[0, 1]) / s
            return jp.array([w, x, y, z])
        
        def case_neg_trace():
            # Find largest diagonal element
            i = jp.argmax(jp.array([mat[0, 0], mat[1, 1], mat[2, 2]]))
            
            def case_0():
                s = jp.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
                return jp.array([
                    (mat[2, 1] - mat[1, 2]) / s,
                    0.25 * s,
                    (mat[0, 1] + mat[1, 0]) / s,
                    (mat[0, 2] + mat[2, 0]) / s,
                ])
            
            def case_1():
                s = jp.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
                return jp.array([
                    (mat[0, 2] - mat[2, 0]) / s,
                    (mat[0, 1] + mat[1, 0]) / s,
                    0.25 * s,
                    (mat[1, 2] + mat[2, 1]) / s,
                ])
            
            def case_2():
                s = jp.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
                return jp.array([
                    (mat[1, 0] - mat[0, 1]) / s,
                    (mat[0, 2] + mat[2, 0]) / s,
                    (mat[1, 2] + mat[2, 1]) / s,
                    0.25 * s,
                ])
            
            return jax.lax.switch(i, [case_0, case_1, case_2])
        
        return jax.lax.cond(trace > 0, case_pos_trace, case_neg_trace)
    
    def _create_initial_data(
        self,
        rng: jax.Array,
        arm_qpos: Optional[jax.Array] = None,
        gripper_open: float = 0.04,
    ) -> mjx.Data:
        """Create initial MJX data with given arm configuration.
        
        Args:
            rng: Random key for noise
            arm_qpos: Initial arm joint positions (7 values), uses home if None
            gripper_open: Gripper opening (0=closed, 0.04=open)
            
        Returns:
            Initialized MJX Data
        """
        if arm_qpos is None:
            arm_qpos = PANDA_HOME_QPOS
            
        # Add small noise to arm positions
        rng, rng_noise = jax.random.split(rng)
        arm_qpos = arm_qpos + jax.random.uniform(
            rng_noise, shape=(7,), minval=-0.05, maxval=0.05
        )
        
        # Clamp to joint limits
        arm_qpos = jp.clip(arm_qpos, PANDA_JOINT_LIMITS[:, 0], PANDA_JOINT_LIMITS[:, 1])
        
        # Build full qpos - use qpos addresses for correct placement
        nq = self._mjx_model.nq
        qpos = jp.zeros(nq)
        
        # Set arm joint positions
        for i, adr in enumerate(self._arm_qposadr):
            qpos = qpos.at[adr].set(arm_qpos[i])
        
        # Set finger positions
        qpos = qpos.at[self._finger_left_qposadr].set(gripper_open)
        qpos = qpos.at[self._finger_right_qposadr].set(gripper_open)
        
        # Initialize velocities to zero
        qvel = jp.zeros(self._mjx_model.nv)
        
        # Initialize control to match initial joint positions
        # This is important for delta control - we start from the initial pose
        nu = self._mjx_model.nu
        ctrl = jp.zeros(nu)
        ctrl = ctrl.at[:7].set(arm_qpos)  # Arm position targets
        ctrl = ctrl.at[7].set(gripper_open)  # Gripper position target
        
        # Create data with initial control
        data = mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        return data
    
    def _apply_action(self, state: State, action: jax.Array) -> mjx.Data:
        """Apply action using delta position control and step physics.
        
        Uses the same approach as mujoco_playground: actions are small deltas
        added to the current control, then clipped to actuator limits.
        
        Args:
            state: Current state
            action: 8D action [7 arm joints, 1 gripper] in [-1, 1]
            
        Returns:
            New MJX Data after physics step
        """
        # Clip action to [-1, 1]
        action = jp.clip(action, -1.0, 1.0)
        
        # Delta control: action * scale is added to current ctrl
        # This gives smooth, incremental motion
        delta = action * self._action_scale
        
        # Get current control and add delta
        ctrl = state.data.ctrl + delta
        
        # Clip to actuator control limits
        ctrl = jp.clip(ctrl, self._ctrl_lowers, self._ctrl_uppers)
        
        # Step physics
        data = mjx_env.step(
            self._mjx_model, state.data, ctrl, self.n_substeps
        )
        
        # Forward kinematics to update site positions
        data = mjx.forward(self._mjx_model, data)
        
        return data
    
    def _check_physics_valid(self, data: mjx.Data) -> jax.Array:
        """Check if physics simulation is valid (no NaN/Inf)."""
        qpos_valid = ~jp.any(jp.isnan(data.qpos)) & ~jp.any(jp.isinf(data.qpos))
        qvel_valid = ~jp.any(jp.isnan(data.qvel)) & ~jp.any(jp.isinf(data.qvel))
        return qpos_valid & qvel_valid

"""Video rendering utilities for policy visualization."""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import mediapy as media
import mujoco
from mujoco import mjx

from envs.tasks import get_task_env
from envs.multitask_wrapper import create_multitask_env
from multitask_ppo import MultiTaskPPONetworks, make_inference_fn


def render_single_frame(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    width: int = 640,
    height: int = 480,
    camera: str = "track",
) -> np.ndarray:
    """Render a single frame from MuJoCo.
    
    Args:
        mj_model: MuJoCo model
        mj_data: MuJoCo data state
        width: Image width
        height: Image height
        camera: Camera name or ID
        
    Returns:
        RGB image as numpy array [H, W, 3]
    """
    renderer = mujoco.Renderer(mj_model, width=width, height=height)
    renderer.update_scene(mj_data)
    return renderer.render()


def rollout_policy(
    env,
    policy_fn,
    params: Dict[str, Any],
    obs_mean: jax.Array,
    obs_var: jax.Array,
    rng: jax.Array,
    num_steps: int = 500,
    task_id: Optional[int] = None,
) -> Tuple[List[Any], List[float], Dict[str, Any]]:
    """Execute policy rollout and collect states.
    
    Args:
        env: Environment instance
        policy_fn: Policy function (params, obs) -> action
        params: Network parameters
        obs_mean: Observation mean for normalization
        obs_var: Observation variance for normalization
        rng: Random key
        num_steps: Maximum rollout length
        task_id: Specific task ID (for multi-task env)
        
    Returns:
        states: List of environment states
        rewards: List of rewards
        info: Rollout info (success, etc.)
    """
    rng, rng_reset = jax.random.split(rng)
    
    # Reset environment
    if task_id is not None and hasattr(env, "reset_task"):
        state = env.reset_task(rng_reset, task_id)
    else:
        state = env.reset(rng_reset)
    
    states = [state]
    rewards = []
    total_reward = 0.0
    success = False
    
    for step in range(num_steps):
        rng, rng_action = jax.random.split(rng)
        
        # Normalize observation
        obs = state.obs
        obs_normalized = (obs - obs_mean) / (jnp.sqrt(obs_var) + 1e-8)
        
        # Get action from policy
        action = policy_fn(params, obs_normalized, rng_action)
        
        # Step environment
        state = env.step(state, action)
        states.append(state)
        rewards.append(float(state.reward))
        total_reward += float(state.reward)
        
        # Check for success
        if "success" in state.metrics:
            if float(state.metrics["success"]) > 0.5:
                success = True
        
        # Check for done
        if float(state.done) > 0.5:
            break
    
    info = {
        "total_reward": total_reward,
        "num_steps": len(rewards),
        "success": success,
    }
    
    return states, rewards, info


def render_rollouts(
    task_names: Sequence[str],
    params: Dict[str, Any],
    networks: MultiTaskPPONetworks,
    output_dir: str,
    num_episodes: int = 3,
    video_length: int = 500,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> Dict[str, List[str]]:
    """Render rollout videos for each task.
    
    Args:
        task_names: Names of tasks to render
        params: Trained network parameters
        networks: PPO networks
        output_dir: Directory to save videos
        num_episodes: Number of episodes per task
        video_length: Maximum frames per video
        width: Video width
        height: Video height
        fps: Video frame rate
        
    Returns:
        Dict mapping task names to list of video paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create inference function
    inference_fn = make_inference_fn(networks)
    
    # Extract normalization stats
    obs_mean = params.get("obs_mean", jnp.zeros(100))
    obs_var = params.get("obs_var", jnp.ones(100))
    network_params = params["params"]
    
    video_paths = {}
    rng = jax.random.PRNGKey(0)
    
    for task_name in task_names:
        print(f"Rendering {task_name}...")
        task_videos = []
        
        # Get task environment
        env = get_task_env(task_name)
        
        for ep in range(num_episodes):
            rng, rng_rollout = jax.random.split(rng)
            
            # Collect rollout
            states, rewards, info = rollout_policy(
                env=env,
                policy_fn=inference_fn,
                params=network_params,
                obs_mean=obs_mean[:env.observation_size],
                obs_var=obs_var[:env.observation_size],
                rng=rng_rollout,
                num_steps=video_length,
            )
            
            # Render frames
            frames = []
            mj_data = mujoco.MjData(env._mj_model)
            
            for state in states:
                # Copy MJX state to MuJoCo
                mjx_data = state.pipeline_state
                mj_data.qpos[:] = np.array(mjx_data.qpos)
                mj_data.qvel[:] = np.array(mjx_data.qvel)
                mujoco.mj_forward(env._mj_model, mj_data)
                
                frame = render_single_frame(
                    env._mj_model, mj_data, width, height
                )
                frames.append(frame)
            
            # Save video
            video_path = os.path.join(
                output_dir, f"{task_name}_ep{ep}_r{info['total_reward']:.1f}.mp4"
            )
            media.write_video(video_path, frames, fps=fps)
            task_videos.append(video_path)
            
            status = "✓ Success" if info["success"] else "✗ Failed"
            print(f"  Episode {ep + 1}: {status}, Reward: {info['total_reward']:.2f}")
        
        video_paths[task_name] = task_videos
    
    return video_paths


def render_multitask_rollouts(
    params: Dict[str, Any],
    networks: MultiTaskPPONetworks,
    task_names: Sequence[str],
    output_dir: str,
    num_episodes: int = 2,
    video_length: int = 500,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> Dict[str, List[str]]:
    """Render rollouts using the multi-task policy.
    
    Args:
        params: Trained multi-task parameters
        networks: Multi-task PPO networks
        task_names: Tasks to render
        output_dir: Output directory
        num_episodes: Episodes per task
        video_length: Max frames
        width: Video width
        height: Video height
        fps: Frame rate
        
    Returns:
        Dict mapping task names to video paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create multi-task environment
    env = create_multitask_env(task_names=task_names)
    
    # Inference function
    inference_fn = make_inference_fn(networks)
    
    obs_mean = params.get("obs_mean", jnp.zeros(env.observation_size))
    obs_var = params.get("obs_var", jnp.ones(env.observation_size))
    network_params = params["params"]
    
    video_paths = {}
    rng = jax.random.PRNGKey(42)
    
    for task_idx, task_name in enumerate(task_names):
        print(f"Rendering multi-task policy on {task_name}...")
        task_videos = []
        
        # Get single-task env for rendering
        single_env = get_task_env(task_name)
        
        for ep in range(num_episodes):
            rng, rng_rollout = jax.random.split(rng)
            
            # Reset to specific task
            state = env.reset_task(rng_rollout, task_idx)
            
            frames = []
            total_reward = 0.0
            success = False
            
            mj_data = mujoco.MjData(single_env._mj_model)
            
            for step in range(video_length):
                rng, rng_action = jax.random.split(rng)
                
                # Normalize and get action
                obs_normalized = (state.obs - obs_mean) / (jnp.sqrt(obs_var) + 1e-8)
                action = inference_fn(network_params, obs_normalized, rng_action)
                
                # Render current state
                mjx_data = state.pipeline_state
                mj_data.qpos[:] = np.array(mjx_data.qpos)
                mj_data.qvel[:] = np.array(mjx_data.qvel)
                mujoco.mj_forward(single_env._mj_model, mj_data)
                
                frame = render_single_frame(single_env._mj_model, mj_data, width, height)
                frames.append(frame)
                
                # Step
                state = env.step(state, action)
                total_reward += float(state.reward)
                
                if "success" in state.metrics and float(state.metrics["success"]) > 0.5:
                    success = True
                
                if float(state.done) > 0.5:
                    break
            
            # Save video
            video_path = os.path.join(
                output_dir, f"multitask_{task_name}_ep{ep}_r{total_reward:.1f}.mp4"
            )
            media.write_video(video_path, frames, fps=fps)
            task_videos.append(video_path)
            
            status = "✓" if success else "✗"
            print(f"  Episode {ep + 1}: {status} Reward: {total_reward:.2f}")
        
        video_paths[task_name] = task_videos
    
    return video_paths


def create_comparison_video(
    multitask_videos: Dict[str, str],
    baseline_videos: Dict[str, str],
    output_path: str,
    task_names: Sequence[str],
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> str:
    """Create side-by-side comparison video.
    
    Args:
        multitask_videos: Dict of task -> video path (multi-task)
        baseline_videos: Dict of task -> video path (baseline)
        output_path: Output video path
        task_names: Tasks to include
        width: Single video width
        height: Single video height
        fps: Frame rate
        
    Returns:
        Path to comparison video
    """
    comparison_frames = []
    
    for task_name in task_names:
        if task_name not in multitask_videos or task_name not in baseline_videos:
            continue
        
        mt_frames = media.read_video(multitask_videos[task_name])
        bl_frames = media.read_video(baseline_videos[task_name])
        
        # Ensure same length
        max_len = max(len(mt_frames), len(bl_frames))
        
        for i in range(max_len):
            mt_frame = mt_frames[min(i, len(mt_frames) - 1)]
            bl_frame = bl_frames[min(i, len(bl_frames) - 1)]
            
            # Add labels
            import cv2
            mt_frame = cv2.putText(
                mt_frame.copy(), "Multi-Task",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            bl_frame = cv2.putText(
                bl_frame.copy(), "Baseline",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            
            # Concatenate side by side
            combined = np.concatenate([mt_frame, bl_frame], axis=1)
            comparison_frames.append(combined)
    
    media.write_video(output_path, comparison_frames, fps=fps)
    print(f"Saved comparison video to {output_path}")
    
    return output_path


def create_task_montage(
    video_paths: Dict[str, str],
    output_path: str,
    task_names: Sequence[str],
    grid_size: Tuple[int, int] = (2, 2),
    width: int = 320,
    height: int = 240,
    fps: int = 30,
) -> str:
    """Create a montage video showing all tasks simultaneously.
    
    Args:
        video_paths: Dict of task -> video path
        output_path: Output path
        task_names: Tasks to include
        grid_size: (rows, cols) for montage
        width: Single tile width
        height: Single tile height
        fps: Frame rate
        
    Returns:
        Path to montage video
    """
    import cv2
    
    rows, cols = grid_size
    
    # Load all videos
    all_frames = {}
    max_length = 0
    
    for task_name in task_names[:rows * cols]:
        if task_name in video_paths:
            frames = media.read_video(video_paths[task_name])
            # Resize frames
            frames = [cv2.resize(f, (width, height)) for f in frames]
            all_frames[task_name] = frames
            max_length = max(max_length, len(frames))
    
    # Create montage frames
    montage_frames = []
    
    for i in range(max_length):
        grid = []
        
        for r in range(rows):
            row_frames = []
            for c in range(cols):
                task_idx = r * cols + c
                if task_idx < len(task_names):
                    task_name = task_names[task_idx]
                    if task_name in all_frames:
                        frames = all_frames[task_name]
                        frame = frames[min(i, len(frames) - 1)]
                        
                        # Add task label
                        frame = cv2.putText(
                            frame.copy(),
                            task_name.replace("_", " ").title(),
                            (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                        row_frames.append(frame)
                    else:
                        row_frames.append(np.zeros((height, width, 3), dtype=np.uint8))
                else:
                    row_frames.append(np.zeros((height, width, 3), dtype=np.uint8))
            
            grid.append(np.concatenate(row_frames, axis=1))
        
        montage = np.concatenate(grid, axis=0)
        montage_frames.append(montage)
    
    media.write_video(output_path, montage_frames, fps=fps)
    print(f"Saved task montage to {output_path}")
    
    return output_path

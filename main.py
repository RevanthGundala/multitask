"""Multi-Task Manipulation with Shared Representation Learning.

A demonstration of training a single policy to solve multiple manipulation
tasks using a shared trunk with task-specific heads, compared against
single-task baselines.

Usage:
    # Train multi-task policy
    python main.py train --mode multitask
    
    # Train baseline policies (one per task)
    python main.py train --mode baseline
    
    # Run both for comparison
    python main.py train --mode both
    
    # Monitor training with TensorBoard
    python main.py tensorboard
    
    # Compare trained models
    python main.py compare --multitask-dir logs/multitask_ppo_* --baseline-dir logs/baseline_ppo_*
    
    # Render videos
    python main.py render --checkpoint logs/multitask_ppo_*/checkpoints/final
    
    # Quick demo (reduced training)
    python main.py demo
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_jax():
    """Configure JAX for optimal GPU usage."""
    # Enable memory preallocation for better performance
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    
    import jax
    
    # Print device info
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"JAX default backend: {jax.default_backend()}")
    
    return jax


def train_command(args):
    """Handle training commands."""
    from config import get_config, get_baseline_config
    
    # Apply config overrides
    if args.mode in ("multitask", "both"):
        config = get_config()
        
        # Apply overrides
        if args.num_envs:
            config.training.num_envs = args.num_envs
        if args.num_timesteps:
            config.training.num_timesteps = args.num_timesteps
        if args.seed:
            config.training.seed = args.seed
        if args.tasks:
            config.env.tasks = tuple(args.tasks)
        
        print("\n" + "=" * 60)
        print("MULTI-TASK TRAINING")
        print("=" * 60)
        print(f"Tasks: {config.env.tasks}")
        print(f"Timesteps: {config.training.num_timesteps:,}")
        print(f"Parallel envs: {config.training.num_envs}")
        
        from train_multitask import train_multitask
        multitask_params, multitask_metrics = train_multitask(config)
        
        # Save metrics
        import json
        log_dir = Path(config.logging.log_dir)
        latest_dir = sorted(log_dir.glob(f"{config.logging.experiment_name}_*"))[-1]
        with open(latest_dir / "metrics.json", "w") as f:
            json.dump(multitask_metrics, f)
    
    if args.mode in ("baseline", "both"):
        config = get_baseline_config()
        
        if args.num_envs:
            config.training.num_envs = args.num_envs
        if args.seed:
            config.training.seed = args.seed
        if args.tasks:
            config.env.tasks = tuple(args.tasks)
        
        print("\n" + "=" * 60)
        print("BASELINE TRAINING (Separate policies)")
        print("=" * 60)
        
        from train_baseline import train_baseline
        baseline_results = train_baseline(config)
        
        # Save metrics
        log_dir = Path(config.logging.log_dir)
        latest_dir = sorted(log_dir.glob(f"{config.logging.experiment_name}_*"))[-1]
        for task_name, (_, metrics) in baseline_results.items():
            with open(latest_dir / task_name / "metrics.json", "w") as f:
                import json
                json.dump(metrics, f)


def compare_command(args):
    """Handle comparison commands."""
    from evaluation.compare import compare_experiments
    from config import get_config
    
    config = get_config()
    task_names = list(config.env.tasks)
    
    if not args.multitask_dir or not args.baseline_dir:
        # Find latest logs
        log_dir = Path(config.logging.log_dir)
        
        if not args.multitask_dir:
            mt_dirs = sorted(log_dir.glob("multitask_ppo_*"))
            if mt_dirs:
                args.multitask_dir = str(mt_dirs[-1])
            else:
                print("Error: No multi-task logs found. Run training first.")
                return
        
        if not args.baseline_dir:
            bl_dirs = sorted(log_dir.glob("baseline_ppo_*"))
            if bl_dirs:
                args.baseline_dir = str(bl_dirs[-1])
            else:
                print("Error: No baseline logs found. Run training first.")
                return
    
    output_dir = args.output_dir or os.path.join(config.logging.log_dir, "comparison")
    
    print(f"\nComparing experiments:")
    print(f"  Multi-task: {args.multitask_dir}")
    print(f"  Baseline: {args.baseline_dir}")
    print(f"  Output: {output_dir}")
    
    summary = compare_experiments(
        multitask_log_dir=args.multitask_dir,
        baseline_log_dir=args.baseline_dir,
        task_names=task_names,
        output_dir=output_dir,
    )
    
    return summary


def render_command(args):
    """Handle rendering commands."""
    from config import get_config
    import orbax.checkpoint as ocp
    
    config = get_config()
    
    if not args.checkpoint:
        # Find latest checkpoint
        log_dir = Path(config.logging.log_dir)
        mt_dirs = sorted(log_dir.glob("multitask_ppo_*"))
        if mt_dirs:
            checkpoint_dirs = sorted((mt_dirs[-1] / "checkpoints").glob("step_*"))
            if checkpoint_dirs:
                args.checkpoint = str(checkpoint_dirs[-1])
    
    if not args.checkpoint:
        print("Error: No checkpoint found. Specify with --checkpoint or train first.")
        return
    
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Load parameters
    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(args.checkpoint)
    
    # Create networks
    from multitask_ppo import make_multitask_ppo_networks
    from envs.multitask_wrapper import create_multitask_env
    
    env = create_multitask_env(task_names=config.env.tasks)
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
    
    # Render videos
    from evaluation.render import render_multitask_rollouts, create_task_montage
    
    output_dir = args.output_dir or os.path.join(config.logging.log_dir, "videos")
    
    video_paths = render_multitask_rollouts(
        params=params,
        networks=networks,
        task_names=config.env.tasks,
        output_dir=output_dir,
        num_episodes=args.num_episodes,
        video_length=config.eval.video_length,
        width=config.eval.render_width,
        height=config.eval.render_height,
    )
    
    # Create montage
    first_videos = {task: paths[0] for task, paths in video_paths.items() if paths}
    create_task_montage(
        video_paths=first_videos,
        output_path=os.path.join(output_dir, "all_tasks_montage.mp4"),
        task_names=list(config.env.tasks),
    )
    
    print(f"\nVideos saved to: {output_dir}")


def demo_command(args):
    """Run a quick demonstration with reduced training."""
    from config import get_config
    
    config = get_config()
    
    # Very reduced settings for demo (especially for CPU testing)
    config.training.num_timesteps = 20_000  # Very small for quick test
    config.training.num_envs = 32  # Much fewer envs for faster JIT
    config.training.log_frequency = 1
    config.ppo.unroll_length = 10  # Shorter rollouts
    config.ppo.num_minibatches = 2  # Fewer minibatches
    config.env.tasks = ("reach",)  # Just 1 task for speed
    
    print("\n" + "=" * 60)
    print("DEMO MODE (Reduced training)")
    print("=" * 60)
    print(f"Tasks: {config.env.tasks}")
    print(f"Timesteps: {config.training.num_timesteps:,}")
    print(f"Num envs: {config.training.num_envs}")
    print("This should take ~1-2 minutes on CPU")
    print("=" * 60 + "\n")
    
    # from train_multitask import train_multitask
    # params, metrics = train_multitask(config)
    from train_baseline import train_baseline
    baseline_results = train_baseline(config)
    
    print("\n✓ Demo training complete!")
    print("To train the full model, run: python main.py train --mode multitask")


def tensorboard_command(args):
    """Launch TensorBoard to visualize training progress."""
    import subprocess
    import webbrowser
    from config import get_config
    
    config = get_config()
    log_dir = args.logdir or config.logging.log_dir
    port = args.port or 6006
    
    print(f"\n🚀 Starting TensorBoard...")
    print(f"   Log directory: {log_dir}")
    print(f"   URL: http://localhost:{port}")
    print("\nPress Ctrl+C to stop.\n")
    
    # Open browser after a short delay
    if not args.no_browser:
        import threading
        def open_browser():
            import time
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Run tensorboard
    subprocess.run([
        "tensorboard",
        "--logdir", log_dir,
        "--port", str(port),
        "--bind_all" if args.bind_all else "--host=localhost",
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Task Manipulation with Shared Representation Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train policies")
    train_parser.add_argument(
        "--mode",
        choices=["multitask", "baseline", "both"],
        default="both",
        help="Training mode",
    )
    train_parser.add_argument(
        "--num-envs",
        type=int,
        help="Number of parallel environments",
    )
    train_parser.add_argument(
        "--num-timesteps",
        type=int,
        help="Total training timesteps",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    train_parser.add_argument(
        "--tasks",
        nargs="+",
        help="Task names to train on",
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument(
        "--multitask-dir",
        help="Multi-task experiment log directory",
    )
    compare_parser.add_argument(
        "--baseline-dir",
        help="Baseline experiment log directory",
    )
    compare_parser.add_argument(
        "--output-dir",
        help="Output directory for comparison plots",
    )
    
    # Render command
    render_parser = subparsers.add_parser("render", help="Render policy videos")
    render_parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory",
    )
    render_parser.add_argument(
        "--output-dir",
        help="Output directory for videos",
    )
    render_parser.add_argument(
        "--num-episodes",
        type=int,
        default=2,
        help="Number of episodes per task",
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run quick demo")
    
    # TensorBoard command
    tb_parser = subparsers.add_parser("tensorboard", help="Launch TensorBoard")
    tb_parser.add_argument(
        "--logdir",
        help="Log directory to visualize (default: logs/)",
    )
    tb_parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port for TensorBoard server",
    )
    tb_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    tb_parser.add_argument(
        "--bind-all",
        action="store_true",
        help="Allow external connections (for remote servers)",
    )
    
    args = parser.parse_args()
    
    # Setup JAX
    setup_jax()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "compare":
        compare_command(args)
    elif args.command == "render":
        render_command(args)
    elif args.command == "demo":
        demo_command(args)
    elif args.command == "tensorboard":
        tensorboard_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

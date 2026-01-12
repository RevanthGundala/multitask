"""Comparison and plotting utilities for multi-task vs baseline experiments."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_metrics(log_dir: str) -> List[Dict[str, Any]]:
    """Load training metrics from a log directory.
    
    Looks for metrics.json or parses TensorBoard logs.
    """
    metrics_file = os.path.join(log_dir, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    
    # Try to load from TensorBoard events
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        metrics = []
        tags = ea.Tags()["scalars"]
        
        if "reward/mean" in tags:
            for event in ea.Scalars("reward/mean"):
                metrics.append({
                    "step": event.step,
                    "reward/mean": event.value,
                })
        
        return metrics
    except Exception as e:
        print(f"Warning: Could not load metrics from {log_dir}: {e}")
        return []


def smooth_curve(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def plot_learning_curves(
    multitask_metrics: List[Dict[str, Any]],
    baseline_metrics: Dict[str, List[Dict[str, Any]]],
    task_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Multi-Task vs Baseline Learning Curves",
) -> plt.Figure:
    """Plot learning curves comparing multi-task to baselines.
    
    Args:
        multitask_metrics: Metrics from multi-task training
        baseline_metrics: Dict mapping task name to baseline metrics
        task_names: List of task names
        output_path: Path to save figure (if provided)
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(task_names)))
    
    for idx, task_name in enumerate(task_names):
        ax = axes[idx]
        
        # Multi-task performance on this task
        mt_steps = [m["step"] for m in multitask_metrics]
        mt_key = f"reward/{task_name}"
        if mt_key in multitask_metrics[0]:
            mt_rewards = [m[mt_key] for m in multitask_metrics]
            mt_rewards_smooth = smooth_curve(np.array(mt_rewards))
            ax.plot(
                mt_steps, mt_rewards_smooth,
                color=colors[idx], linewidth=2,
                label="Multi-Task",
            )
            ax.fill_between(
                mt_steps,
                mt_rewards_smooth - np.std(mt_rewards) * 0.5,
                mt_rewards_smooth + np.std(mt_rewards) * 0.5,
                color=colors[idx], alpha=0.2,
            )
        
        # Baseline performance
        if task_name in baseline_metrics:
            bl_metrics = baseline_metrics[task_name]
            bl_steps = [m["step"] for m in bl_metrics]
            bl_rewards = [m["reward/mean"] for m in bl_metrics]
            bl_rewards_smooth = smooth_curve(np.array(bl_rewards))
            ax.plot(
                bl_steps, bl_rewards_smooth,
                color="gray", linewidth=2, linestyle="--",
                label="Baseline (Single-Task)",
            )
        
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episode Reward")
        ax.set_title(f"Task: {task_name.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved learning curves to {output_path}")
    
    return fig


def compute_sample_efficiency(
    multitask_metrics: List[Dict[str, Any]],
    baseline_metrics: Dict[str, List[Dict[str, Any]]],
    task_names: List[str],
    reward_threshold: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """Compute sample efficiency metrics.
    
    Sample efficiency = steps to reach threshold reward.
    
    Args:
        multitask_metrics: Multi-task training metrics
        baseline_metrics: Baseline metrics per task
        task_names: Task names
        reward_threshold: Threshold to measure (default: 0 = positive reward)
        
    Returns:
        Dict with sample efficiency statistics per task
    """
    results = {}
    
    for task_name in task_names:
        task_results = {
            "multitask_steps": None,
            "baseline_steps": None,
            "efficiency_ratio": None,
        }
        
        # Multi-task steps to threshold
        mt_key = f"reward/{task_name}"
        for m in multitask_metrics:
            if mt_key in m and m[mt_key] >= reward_threshold:
                task_results["multitask_steps"] = m["step"]
                break
        
        # Baseline steps to threshold
        if task_name in baseline_metrics:
            for m in baseline_metrics[task_name]:
                if m["reward/mean"] >= reward_threshold:
                    task_results["baseline_steps"] = m["step"]
                    break
        
        # Efficiency ratio (baseline / multitask)
        if task_results["multitask_steps"] and task_results["baseline_steps"]:
            task_results["efficiency_ratio"] = (
                task_results["baseline_steps"] / task_results["multitask_steps"]
            )
        
        results[task_name] = task_results
    
    return results


def plot_sample_efficiency(
    efficiency_results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot sample efficiency comparison bar chart.
    
    Args:
        efficiency_results: Output from compute_sample_efficiency
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tasks = list(efficiency_results.keys())
    x = np.arange(len(tasks))
    width = 0.35
    
    mt_steps = [
        efficiency_results[t]["multitask_steps"] or 0 
        for t in tasks
    ]
    bl_steps = [
        efficiency_results[t]["baseline_steps"] or 0 
        for t in tasks
    ]
    
    # Normalize to millions
    mt_steps_m = [s / 1e6 for s in mt_steps]
    bl_steps_m = [s / 1e6 for s in bl_steps]
    
    bars1 = ax.bar(x - width/2, mt_steps_m, width, label="Multi-Task", color="steelblue")
    bars2 = ax.bar(x + width/2, bl_steps_m, width, label="Baseline", color="coral")
    
    ax.set_xlabel("Task")
    ax.set_ylabel("Steps to Threshold (Millions)")
    ax.set_title("Sample Efficiency: Multi-Task vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ").title() for t in tasks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add efficiency ratio labels
    for i, task in enumerate(tasks):
        ratio = efficiency_results[task]["efficiency_ratio"]
        if ratio:
            ax.annotate(
                f"{ratio:.1f}x faster" if ratio > 1 else f"{1/ratio:.1f}x slower",
                xy=(i, max(mt_steps_m[i], bl_steps_m[i])),
                ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved efficiency plot to {output_path}")
    
    return fig


def plot_transfer_analysis(
    multitask_metrics: List[Dict[str, Any]],
    task_names: List[str],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Analyze and plot transfer learning effects.
    
    Shows how training on one task affects others.
    
    Args:
        multitask_metrics: Multi-task training metrics
        task_names: Task names
        output_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = [m["step"] for m in multitask_metrics]
    
    for task_name in task_names:
        key = f"reward/{task_name}"
        if key in multitask_metrics[0]:
            rewards = [m[key] for m in multitask_metrics]
            rewards_smooth = smooth_curve(np.array(rewards), weight=0.95)
            ax.plot(steps, rewards_smooth, linewidth=2, label=task_name.replace("_", " ").title())
    
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Multi-Task Learning: Per-Task Performance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved transfer analysis to {output_path}")
    
    return fig


def compare_experiments(
    multitask_log_dir: str,
    baseline_log_dir: str,
    task_names: List[str],
    output_dir: str,
) -> Dict[str, Any]:
    """Run full comparison between multi-task and baseline experiments.
    
    Args:
        multitask_log_dir: Directory with multi-task logs
        baseline_log_dir: Directory with baseline logs
        task_names: List of task names
        output_dir: Directory to save comparison outputs
        
    Returns:
        Comparison summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics
    print("Loading multi-task metrics...")
    multitask_metrics = load_metrics(multitask_log_dir)
    
    print("Loading baseline metrics...")
    baseline_metrics = {}
    for task_name in task_names:
        task_dir = os.path.join(baseline_log_dir, task_name)
        if os.path.exists(task_dir):
            baseline_metrics[task_name] = load_metrics(task_dir)
    
    # Generate plots
    print("Generating comparison plots...")
    
    plot_learning_curves(
        multitask_metrics, baseline_metrics, task_names,
        output_path=os.path.join(output_dir, "learning_curves.png"),
    )
    
    efficiency_results = compute_sample_efficiency(
        multitask_metrics, baseline_metrics, task_names,
    )
    
    plot_sample_efficiency(
        efficiency_results,
        output_path=os.path.join(output_dir, "sample_efficiency.png"),
    )
    
    plot_transfer_analysis(
        multitask_metrics, task_names,
        output_path=os.path.join(output_dir, "transfer_analysis.png"),
    )
    
    # Compute summary statistics
    summary = {
        "sample_efficiency": efficiency_results,
        "final_rewards": {},
    }
    
    # Final rewards
    if multitask_metrics:
        final_mt = multitask_metrics[-1]
        for task_name in task_names:
            key = f"reward/{task_name}"
            if key in final_mt:
                summary["final_rewards"][f"{task_name}_multitask"] = final_mt[key]
    
    for task_name, bl_metrics in baseline_metrics.items():
        if bl_metrics:
            summary["final_rewards"][f"{task_name}_baseline"] = bl_metrics[-1]["reward/mean"]
    
    # Total training time comparison
    total_mt_steps = multitask_metrics[-1]["step"] if multitask_metrics else 0
    total_bl_steps = sum(
        bl[-1]["step"] if bl else 0 
        for bl in baseline_metrics.values()
    )
    
    summary["total_steps"] = {
        "multitask": total_mt_steps,
        "baseline_combined": total_bl_steps,
        "step_ratio": total_bl_steps / total_mt_steps if total_mt_steps > 0 else 0,
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        # Convert numpy types to Python types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer, jnp.integer)):
                return int(obj)
            if isinstance(obj, (np.floating, jnp.floating)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        json.dump(convert(summary), f, indent=2)
    
    print(f"\nComparison summary saved to {summary_path}")
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Multi-task total steps: {total_mt_steps:,}")
    print(f"Baseline combined steps: {total_bl_steps:,}")
    print(f"Step efficiency ratio: {summary['total_steps']['step_ratio']:.2f}x")
    
    return summary


def create_summary_table(
    multitask_metrics: List[Dict[str, Any]],
    baseline_metrics: Dict[str, List[Dict[str, Any]]],
    task_names: List[str],
) -> pd.DataFrame:
    """Create a summary table comparing methods.
    
    Args:
        multitask_metrics: Multi-task metrics
        baseline_metrics: Baseline metrics per task
        task_names: Task names
        
    Returns:
        pandas DataFrame with comparison
    """
    rows = []
    
    for task_name in task_names:
        row = {"Task": task_name.replace("_", " ").title()}
        
        # Multi-task final reward
        mt_key = f"reward/{task_name}"
        if multitask_metrics and mt_key in multitask_metrics[-1]:
            row["Multi-Task (Final)"] = f"{multitask_metrics[-1][mt_key]:.3f}"
        else:
            row["Multi-Task (Final)"] = "N/A"
        
        # Baseline final reward
        if task_name in baseline_metrics and baseline_metrics[task_name]:
            row["Baseline (Final)"] = f"{baseline_metrics[task_name][-1]['reward/mean']:.3f}"
        else:
            row["Baseline (Final)"] = "N/A"
        
        rows.append(row)
    
    return pd.DataFrame(rows)

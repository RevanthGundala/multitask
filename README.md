# Multi-Task Manipulation with Shared Representation Learning

Train a single policy to solve multiple manipulation tasks using a shared trunk with task-specific heads. Demonstrates that shared representations learn faster and generalize better than training separate policies.

## 🎯 Key Features

- **Shared Trunk Architecture**: Common feature extraction across all tasks with task-specific policy/value heads
- **4 Manipulation Tasks**: Reach, Push, Pick-and-Place, Peg Insertion
- **MJX + PPO**: GPU-accelerated physics with vectorized PPO training (2000+ parallel envs)
- **Fair Baseline Comparison**: Same hyperparameters for single-task vs multi-task
- **Comprehensive Evaluation**: Learning curves, sample efficiency, transfer analysis

## 🏗️ Architecture

```
                    Observation (+ task one-hot)
                            │
                            ▼
                ┌───────────────────────┐
                │    Shared Trunk       │
                │  (256 → 256 MLP)      │
                └───────────┬───────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ Policy  │         │  Value  │         │  Task   │
   │  Head   │         │  Head   │         │ Embed   │
   │ (128)   │         │ (128)   │         │ (opt)   │
   └─────────┘         └─────────┘         └─────────┘
        │                   │
        ▼                   ▼
     Actions              Value
```

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Run quick demo (2 tasks, 500K steps, ~5 min on GPU)
uv run python main.py demo

# Full training (4 tasks, 10M steps each mode)
uv run python main.py train --mode both

# Compare results
uv run python main.py compare

# Render policy videos
uv run python main.py render
```

## 📊 Expected Results

| Metric | Multi-Task | Baseline (avg) |
|--------|------------|----------------|
| Sample Efficiency | 1x | ~2-3x more samples |
| Final Performance | Comparable | Comparable |
| Training Time | 1x | 4x (separate policies) |
| Transfer | ✓ Positive | N/A |

## 🔧 Configuration

Edit `configs/default.py` to customize:

```python
config.env.tasks = ("reach", "push", "pick_place", "peg_insert")
config.training.num_timesteps = 10_000_000
config.training.num_envs = 2048
config.network.shared_layer_sizes = (256, 256)
config.network.policy_head_sizes = (128,)
```

## 📁 Project Structure

```
multitask/
├── main.py                 # CLI entry point
├── configs/
│   └── default.py          # Training configuration
├── envs/
│   ├── tasks/              # Individual task environments
│   │   ├── reach.py        # Reach target position
│   │   ├── push.py         # Push object to goal
│   │   ├── pick_place.py   # Pick and place object
│   │   └── peg_insert.py   # Precision peg insertion
│   └── multitask_wrapper.py # Multi-task environment
├── networks/
│   └── multitask_ppo.py    # Shared trunk + task heads
├── training/
│   ├── train_multitask.py  # Multi-task PPO training
│   └── train_baseline.py   # Single-task baselines
└── evaluation/
    ├── compare.py          # Comparison plots
    └── render.py           # Video rendering
```

## 🖥️ Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4090)
- **RAM**: 16GB+ recommended
- **Training Time**: 
  - Demo: ~5 minutes
  - Full (multi-task only): ~30-60 minutes
  - Full (both modes): ~2-3 hours

## 📈 Visualizations

### TensorBoard (Real-time Training)

Monitor training progress in real-time with TensorBoard:

```bash
# Start TensorBoard (in a separate terminal)
uv run tensorboard --logdir logs

# Or specify a specific run
uv run tensorboard --logdir logs/multitask_ppo_2026*
```

Then open http://localhost:6006 in your browser.

**Metrics logged:**
- `reward/mean`: Average episode reward
- `reward/task_*`: Per-task rewards (reach, push, etc.)
- `loss/policy`: Policy loss
- `loss/value`: Value function loss
- `loss/entropy`: Entropy bonus
- `perf/fps`: Training throughput (frames/sec)

### Comparison Plots (After Training)

After training, the `compare` command generates:

1. **Learning Curves** (`learning_curves.png`): Per-task reward over time
2. **Sample Efficiency** (`sample_efficiency.png`): Steps to reach threshold
3. **Transfer Analysis** (`transfer_analysis.png`): Multi-task performance breakdown

The `render` command generates:

1. **Per-task Videos**: Individual task rollouts
2. **Task Montage**: 2x2 grid showing all tasks simultaneously

## 🔬 Research Context

This project demonstrates:

1. **Positive Transfer**: Shared representations accelerate learning on related tasks
2. **Sample Efficiency**: Multi-task training requires fewer total samples than N separate policies
3. **Practical Robotics**: Foundation for general-purpose manipulation systems

## 📜 License

MIT

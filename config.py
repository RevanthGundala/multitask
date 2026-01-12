"""Default configuration for multi-task manipulation experiments."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Returns the default configuration."""
    config = ml_collections.ConfigDict()

    # ==========================================================================
    # Environment Configuration
    # ==========================================================================
    config.env = ml_collections.ConfigDict()
    config.env.tasks = ("reach", "push", "pick_place", "peg_insert")
    config.env.episode_length = 500
    config.env.action_repeat = 1
    config.env.obs_history_length = 1  # Number of stacked observations

    # ==========================================================================
    # Training Configuration
    # ==========================================================================
    config.training = ml_collections.ConfigDict()
    config.training.seed = 42
    config.training.num_timesteps = 10_000_000
    config.training.num_envs = 2048  # Vectorized envs on GPU
    config.training.num_eval_envs = 128
    config.training.normalize_observations = True
    config.training.log_frequency = 10
    config.training.eval_frequency = 100_000
    config.training.checkpoint_frequency = 500_000

    # ==========================================================================
    # PPO Hyperparameters
    # ==========================================================================
    config.ppo = ml_collections.ConfigDict()
    config.ppo.learning_rate = 3e-5  # Reduced to prevent gradient explosion and entropy collapse
    config.ppo.entropy_cost = 0.05  # Increased to maintain exploration and prevent entropy collapse
    config.ppo.discounting = 0.99
    config.ppo.gae_lambda = 0.95
    config.ppo.unroll_length = 20
    config.ppo.batch_size = 256
    config.ppo.num_minibatches = 8
    config.ppo.num_updates_per_batch = 4
    config.ppo.clip_epsilon = 0.2
    config.ppo.value_loss_coef = 0.5
    config.ppo.max_grad_norm = 0.5
    config.ppo.reward_scaling = 1.0

    # ==========================================================================
    # Network Architecture
    # ==========================================================================
    config.network = ml_collections.ConfigDict()
    # Shared trunk (encoder)
    config.network.shared_layer_sizes = (256, 256)
    # Task-specific heads
    config.network.policy_head_sizes = (128,)
    config.network.value_head_sizes = (128,)
    # Task embedding dimension (0 = one-hot, >0 = learned embedding)
    config.network.task_embedding_dim = 0
    config.network.activation = "swish"

    # ==========================================================================
    # Evaluation & Visualization
    # ==========================================================================
    config.eval = ml_collections.ConfigDict()
    config.eval.num_episodes = 50  # Episodes per task for evaluation
    config.eval.video_length = 500  # Frames per video
    config.eval.render_width = 640
    config.eval.render_height = 480

    # ==========================================================================
    # Logging & Checkpointing
    # ==========================================================================
    config.logging = ml_collections.ConfigDict()
    config.logging.log_dir = "logs"
    config.logging.experiment_name = "multitask_ppo"
    config.logging.use_wandb = False
    config.logging.wandb_project = "multitask-manipulation"

    return config


def get_baseline_config() -> ml_collections.ConfigDict:
    """Returns configuration for baseline (single-task) training."""
    config = get_config()
    config.logging.experiment_name = "baseline_ppo"
    # Baselines train on single task, so we use same total budget per task
    config.training.num_timesteps = 2_500_000  # 10M / 4 tasks
    return config

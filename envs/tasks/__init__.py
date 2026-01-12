"""Task definitions for multi-task manipulation.

All tasks use MuJoCo Playground's MjxEnv base class for consistent
interface and GPU-accelerated simulation.
"""

from typing import Callable, Dict

from mujoco_playground._src.mjx_env import MjxEnv

# Task registry mapping task names to environment constructors
TASK_REGISTRY: Dict[str, Callable[..., MjxEnv]] = {}


def register_task(name: str):
    """Decorator to register a task environment."""
    def decorator(fn: Callable[..., MjxEnv]) -> Callable[..., MjxEnv]:
        TASK_REGISTRY[name] = fn
        return fn
    return decorator


def get_task_env(task_name: str, **kwargs) -> MjxEnv:
    """Get a task environment by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_name](**kwargs)


# Import task modules to trigger registration
from envs.tasks import reach, push, pick_place, peg_insert

__all__ = [
    "TASK_REGISTRY",
    "register_task",
    "get_task_env",
]

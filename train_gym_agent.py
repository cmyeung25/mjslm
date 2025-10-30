"""Simple Q-learning trainer using OpenAI Gym environments.

This script demonstrates how to train a tabular Q-learning agent on
simple, discrete OpenAI Gym environments such as ``FrozenLake-v1`` or
``Taxi-v3``.  The script can be executed directly from the command line
and allows basic configuration through command-line flags.

Example
-------
Run a training session on ``FrozenLake-v1`` with 5000 episodes and save
the learned Q-table::

    python train_gym_agent.py --env FrozenLake-v1 --episodes 5000 --save-path q_table.npy

Requirements
------------
The script depends on ``gym`` and ``numpy``.  Install them via::

    pip install gym numpy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import gym
except ImportError as exc:  # pragma: no cover - guidance for missing dependency
    raise SystemExit(
        "The 'gym' package is required to run this script. Install it via 'pip install gym'."
    ) from exc


def _ensure_directory(path: Path) -> None:
    """Ensure the parent directory of ``path`` exists."""

    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def epsilon_greedy_policy(q_values: np.ndarray, epsilon: float) -> int:
    """Select an action following an epsilon-greedy policy."""

    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    return int(np.argmax(q_values))


def _reset_env(env: gym.Env) -> int:
    """Reset environment and return the initial state index.

    ``gym`` changed its API across versions.  This helper normalises the
    return value to always provide the observation/state as an integer.
    """

    state = env.reset()
    if isinstance(state, tuple):  # gym>=0.26 returns (obs, info)
        state = state[0]
    return int(state)


def _step_env(env: gym.Env, action: int) -> Tuple[int, float, bool]:
    """Take a step in the environment and return ``state, reward, done``."""

    step_result = env.step(action)
    if len(step_result) == 5:  # gym>=0.26 returns (obs, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, _ = step_result
        done = bool(terminated or truncated)
    else:  # legacy API
        next_state, reward, done, _ = step_result
    return int(next_state), float(reward), bool(done)


def _make_env(env_name: str, *, render_mode: str | None = None) -> gym.Env:
    """Create a Gym environment and ensure it is discrete."""

    try:
        env = gym.make(env_name, render_mode=render_mode) if render_mode else gym.make(env_name)
    except TypeError:
        # Some classic control environments (e.g. CartPole-v1) do not accept ``render_mode``
        # until newer Gym versions.  Fall back to the default constructor when unsupported.
        env = gym.make(env_name)

    if not hasattr(env.observation_space, "n") or not hasattr(env.action_space, "n"):
        env.close()
        raise ValueError(
            "This training script only supports environments with discrete observation and action spaces."
        )

    return env


@dataclass
class TrainingLog:
    """Container for per-episode training information."""

    rewards: list[float]
    episode_lengths: list[int]
    successes: list[bool]


def train(
    env_name: str,
    episodes: int = 5000,
    max_steps: int = 100,
    learning_rate: float = 0.1,
    discount: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
) -> Tuple[np.ndarray, TrainingLog]:
    """Train a Q-learning agent on the given environment.

    Parameters
    ----------
    env_name:
        Name of the Gym environment, e.g. ``"FrozenLake-v1"``.
    episodes:
        Number of episodes to train for.
    max_steps:
        Maximum number of steps per episode.
    learning_rate:
        Q-table learning rate (``alpha``).
    discount:
        Discount factor (``gamma``).
    epsilon:
        Initial exploration rate for epsilon-greedy policy.
    epsilon_min:
        Minimum exploration rate.
    epsilon_decay:
        Multiplicative decay applied to ``epsilon`` after each episode.

    Returns
    -------
    A tuple ``(q_table, log)`` where ``q_table`` contains the learned
    action values and ``log`` stores per-episode rewards, episode
    lengths, and a success flag indicating whether the environment's
    goal was achieved.
    """

    env = _make_env(env_name)

    num_states = int(env.observation_space.n)
    num_actions = int(env.action_space.n)

    q_table = np.zeros((num_states, num_actions), dtype=np.float32)
    rewards: list[float] = []
    episode_lengths: list[int] = []
    successes: list[bool] = []

    for episode in range(episodes):
        state = _reset_env(env)
        total_reward = 0.0
        steps_taken = 0

        for step in range(max_steps):
            action = epsilon_greedy_policy(q_table[state], epsilon)
            next_state, reward, done = _step_env(env, action)

            best_next_action = np.max(q_table[next_state])
            td_target = reward + discount * best_next_action
            td_error = td_target - q_table[state, action]
            q_table[state, action] += learning_rate * td_error

            state = next_state
            total_reward += reward
            steps_taken = step + 1

            if done:
                break

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards.append(total_reward)
        episode_lengths.append(steps_taken)
        successes.append(total_reward > 0.0)

    env.close()
    return q_table, TrainingLog(rewards=rewards, episode_lengths=episode_lengths, successes=successes)


def summarise_training(log: TrainingLog, window: int = 100) -> dict:
    """Calculate summary statistics for training history.

    The resulting dictionary exposes aggregate reward metrics, episode
    lengths, an estimated success rate (percentage of episodes ending
    with a positive reward), and a short natural-language interpretation
    to make it easier to reason about how the agent performed.
    """

    rewards_array = np.array(log.rewards, dtype=np.float32)
    lengths_array = np.array(log.episode_lengths, dtype=np.float32)
    successes_array = np.array(log.successes, dtype=np.float32)
    moving_avg = (
        np.convolve(rewards_array, np.ones(window) / window, mode="valid")
        if rewards_array.size >= window
        else np.array([], dtype=np.float32)
    )

    success_rate = float(successes_array.mean()) if successes_array.size else math.nan
    mean_length = float(lengths_array.mean()) if lengths_array.size else math.nan

    return {
        "episodes": int(rewards_array.size),
        "mean_reward": float(rewards_array.mean()) if rewards_array.size else math.nan,
        "max_reward": float(rewards_array.max()) if rewards_array.size else math.nan,
        "mean_episode_length": mean_length,
        "success_rate": success_rate,
        "success_count": int(successes_array.sum()) if successes_array.size else 0,
        "moving_average_window": int(window),
        "moving_average": moving_avg.tolist(),
        "interpretation": _interpret_training_summary(
            episodes=int(rewards_array.size),
            mean_reward=float(rewards_array.mean()) if rewards_array.size else math.nan,
            success_rate=success_rate,
            mean_length=mean_length,
        ),
    }


def _interpret_training_summary(
    *, episodes: int, mean_reward: float, success_rate: float, mean_length: float
) -> str:
    """Create a lightweight natural-language interpretation of training metrics."""

    if episodes == 0:
        return "No training episodes were run."

    parts = [f"Trained for {episodes} episodes."]

    if not math.isnan(mean_reward):
        parts.append(f"Average reward was {mean_reward:.2f} per episode.")

    if not math.isnan(success_rate):
        parts.append(f"The agent reached the goal in approximately {success_rate * 100:.1f}% of attempts.")

    if not math.isnan(mean_length):
        parts.append(f"Episodes lasted about {mean_length:.1f} steps on average.")

    return " ".join(parts)


def evaluate_policy(
    env_name: str,
    q_table: np.ndarray,
    episodes: int,
    max_steps: int,
    *,
    render: bool = False,
) -> dict:
    """Run greedy episodes using the learned Q-table and collect metrics.

    Alongside basic reward statistics the return value includes the
    average number of steps per episode and how frequently the agent
    achieved a positive reward when acting greedily.
    """

    if episodes <= 0:
        return {"episodes": 0, "mean_reward": math.nan, "rewards": []}

    env = _make_env(env_name, render_mode="human" if render else None)

    rewards = []
    episode_lengths = []
    success_count = 0
    num_states = q_table.shape[0]

    for episode in range(episodes):
        state = _reset_env(env)
        total_reward = 0.0
        steps_taken = 0

        for _ in range(max_steps):
            if state >= num_states:
                raise ValueError(
                    "Observed state index is out of bounds for the learned Q-table. "
                    "Ensure you evaluate on the same environment you trained on."
                )

            action = int(np.argmax(q_table[state]))
            next_state, reward, done = _step_env(env, action)
            total_reward += reward

            if render:
                env.render()

            state = next_state
            steps_taken += 1
            if done:
                break

        rewards.append(total_reward)
        episode_lengths.append(steps_taken)
        if total_reward > 0.0:
            success_count += 1

    env.close()

    rewards_array = np.array(rewards, dtype=np.float32)
    mean_reward = float(rewards_array.mean()) if episodes else math.nan
    std_reward = float(rewards_array.std(ddof=0)) if episodes else math.nan
    mean_length = float(np.mean(episode_lengths)) if episode_lengths else math.nan
    success_rate = float(success_count / episodes) if episodes else math.nan
    return {
        "episodes": int(episodes),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "rewards": rewards,
        "mean_episode_length": mean_length,
        "success_rate": success_rate,
        "success_count": int(success_count),
        "interpretation": _interpret_evaluation_summary(
            episodes=episodes,
            mean_reward=mean_reward,
            success_rate=success_rate,
            mean_length=mean_length,
        ),
    }


def _interpret_evaluation_summary(
    *, episodes: int, mean_reward: float, success_rate: float, mean_length: float
) -> str:
    """Convert evaluation metrics into an easy-to-read description."""

    if episodes == 0:
        return "No evaluation episodes were run."

    parts = [f"Evaluated greedily for {episodes} episodes."]

    if not math.isnan(mean_reward):
        parts.append(f"Mean reward was {mean_reward:.2f}.")

    if not math.isnan(success_rate):
        parts.append(f"Success rate was about {success_rate * 100:.1f}%.")

    if not math.isnan(mean_length):
        parts.append(f"Episodes averaged {mean_length:.1f} steps.")

    return " ".join(parts)


def _get_action_meanings(env: gym.Env) -> list[str]:
    """Return a list of human-readable action names for the environment."""

    get_meanings = getattr(env.unwrapped, "get_action_meanings", None)
    if callable(get_meanings):
        try:
            return list(map(str, get_meanings()))
        except TypeError:
            pass  # Fall back to numeric labels if the method has an unexpected signature.

    return [str(i) for i in range(int(env.action_space.n))]


def _prompt_for_action(action_meanings: list[str], recommended: int | None = None) -> int:
    """Prompt the user to pick an action from ``action_meanings``.

    The ``recommended`` action (if provided) will be highlighted to make it
    easy to compare the human choice with the greedy agent's preference.
    """

    prompt_lines = ["Choose an action by index:"]
    for idx, meaning in enumerate(action_meanings):
        if idx == recommended:
            prompt_lines.append(f"  [{idx}] {meaning}  <-- agent recommendation")
        else:
            prompt_lines.append(f"  [{idx}] {meaning}")

    prompt_lines.append("Enter selection: ")
    sys.stdout.write("\n".join(prompt_lines))
    sys.stdout.flush()

    while True:
        try:
            choice = input()
        except EOFError:
            raise SystemExit("Interactive play aborted: no input available.") from None

        choice = choice.strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(action_meanings):
                return idx
        print(f"Invalid choice '{choice}'. Please enter a value between 0 and {len(action_meanings) - 1}.")


def interactive_play(
    env_name: str,
    q_table: np.ndarray | None,
    episodes: int,
    max_steps: int,
    *,
    mode: str,
    render: bool,
    pause: float,
) -> dict:
    """Let a human watch or control the environment after training.

    Parameters
    ----------
    env_name:
        Name of the Gym environment to create.
    q_table:
        Learned Q-table. Required for ``mode`` values other than ``"human"``.
    episodes:
        Number of interactive episodes to run.
    max_steps:
        Maximum number of steps per episode.
    mode:
        ``"agent"`` lets the greedy policy act automatically, ``"human"``
        gives full manual control, and ``"compare"`` asks the human to pick
        actions while simultaneously showing the greedy agent's suggestions.
    render:
        Whether to render the environment on each step.
    pause:
        Seconds to wait between steps when rendering automated play.
    """

    if episodes <= 0:
        return {"episodes": 0, "interpretation": "Interactive play skipped."}

    if mode in {"agent", "compare"} and q_table is None:
        raise ValueError("A learned Q-table is required to let the agent act or provide suggestions.")

    env = _make_env(env_name, render_mode="human" if render else None)
    action_meanings = _get_action_meanings(env)
    num_states = q_table.shape[0] if q_table is not None else int(env.observation_space.n)

    episode_outcomes: list[dict] = []

    for episode in range(episodes):
        state = _reset_env(env)
        total_reward = 0.0
        steps_taken = 0
        if render:
            env.render()

        for step in range(max_steps):
            if state >= num_states:
                raise ValueError(
                    "Observed state index is out of bounds for the learned Q-table. "
                    "Ensure you are interacting with the same environment that was used for training."
                )

            greedy_action = int(np.argmax(q_table[state])) if q_table is not None else None

            if mode == "human":
                action = _prompt_for_action(action_meanings)
            elif mode == "compare":
                action = _prompt_for_action(action_meanings, greedy_action)
            else:  # mode == "agent"
                action = greedy_action

            next_state, reward, done = _step_env(env, action)
            total_reward += reward
            steps_taken = step + 1

            if render:
                env.render()
                if mode == "agent" and pause > 0:
                    time.sleep(pause)

            state = next_state
            if done:
                break

        episode_outcomes.append(
            {
                "reward": float(total_reward),
                "steps": int(steps_taken),
                "success": bool(total_reward > 0.0),
            }
        )

    env.close()

    rewards = [entry["reward"] for entry in episode_outcomes]
    success_count = sum(1 for entry in episode_outcomes if entry["success"])
    mean_reward = float(np.mean(rewards)) if rewards else math.nan
    success_rate = float(success_count / episodes) if episodes else math.nan
    interpretation = (
        f"Interactive {mode} session completed for {episodes} episode(s)."
        if episodes
        else "Interactive play skipped."
    )

    return {
        "episodes": int(episodes),
        "mode": mode,
        "outcomes": episode_outcomes,
        "mean_reward": mean_reward,
        "success_rate": success_rate,
        "interpretation": interpretation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Q-learning agent on a Gym environment")
    parser.add_argument("--env", default="FrozenLake-v1", help="Gym environment name (default: FrozenLake-v1)")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate (alpha)")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Multiplicative epsilon decay per episode")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path to save the learned Q-table as a NumPy .npy file",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional path to save training statistics as JSON",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Number of evaluation episodes to run greedily after training",
    )
    parser.add_argument(
        "--render-eval",
        action="store_true",
        help="Render evaluation episodes using the environment's built-in renderer",
    )
    parser.add_argument(
        "--play-mode",
        choices=["agent", "human", "compare"],
        default=None,
        help="After training, run interactive play as the agent, as a human, or comparing both",
    )
    parser.add_argument(
        "--play-episodes",
        type=int,
        default=1,
        help="Number of interactive play episodes to run",
    )
    parser.add_argument(
        "--play-pause",
        type=float,
        default=0.5,
        help="Seconds to pause between automated steps when the agent is acting",
    )
    parser.add_argument(
        "--play-render",
        action="store_true",
        help="Render interactive play episodes (recommended for visual environments)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    q_table, training_log = train(
        env_name=args.env,
        episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    summary = summarise_training(training_log)

    evaluation = evaluate_policy(
        env_name=args.env,
        q_table=q_table,
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        render=args.render_eval,
    )

    if evaluation["episodes"]:
        summary["evaluation"] = evaluation

    if args.play_mode is not None:
        play_summary = interactive_play(
            env_name=args.env,
            q_table=q_table,
            episodes=args.play_episodes,
            max_steps=args.max_steps,
            mode=args.play_mode,
            render=args.play_render,
            pause=max(args.play_pause, 0.0),
        )
        summary["interactive_play"] = play_summary

    print(json.dumps(summary, indent=2))

    if args.save_path is not None:
        _ensure_directory(args.save_path)
        np.save(args.save_path, q_table)
        print(f"Saved Q-table to {args.save_path.resolve()}")

    if args.summary_path is not None:
        _ensure_directory(args.summary_path)
        args.summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved training summary to {args.summary_path.resolve()}")


if __name__ == "__main__":
    main()

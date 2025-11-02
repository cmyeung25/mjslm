#!/usr/bin/env python3
"""Self-play training harness for the Hong Kong Mahjong Gym environment.

This script shares a single softmax policy across all four seats.  The policy
is trained with a REINFORCE-style update after every round using the score
deltas produced by :class:`hk_mahjong_full_demo.MahjongGame`.  Non-learning
seats are controlled through the new controller hooks, allowing them to consult
exactly the same policy that the learning seat uses when stepping the
``HongKongMahjongEnv`` environment.

Reward design is intentionally tied to the real Hong Kong scoring model so the
agent experiences the natural trade-off between pushing for more fan and
protecting against dangerous discards.  Because the update consumes each
seat's final score delta, a player that deals into another person's win
(``出衝``) receives the full negative payout, whereas locking in a modest ron or
tsumo after declaring ready (``叫糊``) yields a smaller but safe gain.  The
resulting learning signal lets policies discover that taking an extra draw for
an extra fan can be worse than folding when the wall is thin or the hand is
fragile.

Running the script produces a PNG plot that tracks the moving average of the
scaled score and win rate for seat 1 (East) across training episodes. Start a
default run with::

    python train_hk_mahjong_selfplay.py --episodes 300

Install the required dependencies first if they are not already present::

    pip install gymnasium numpy matplotlib

The plot is written to ``outputs/hk_mahjong_learning_curve.png`` by default and
the terminal shows periodic summaries of the East seat's scaled score delta,
moving-average reward, and win rate.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "matplotlib is required to plot the learning curve. Install it with 'pip install matplotlib'."
    ) from exc
import numpy as np

from hk_mahjong_full_demo import HongKongMahjongEnv

STAGE_SELF_ACTION = 1
STAGE_DISCARD = 2
STAGE_REACTION = 3
ACTION_TYPES = [
    "skip",
    "discard",
    "tsumo",
    "concealed_kong",
    "added_kong",
    "ron",
    "kong",
    "pon",
    "chi",
]
ACTION_TYPE_INDEX = {name: idx for idx, name in enumerate(ACTION_TYPES)}


def build_observation(
    game,
    player_index: int,
    stage_code: int,
    *,
    override_last_discard: Optional[int] = None,
) -> dict:
    """Return an observation dictionary mirroring ``HongKongMahjongEnv``."""

    player = game.players[player_index]
    hand_vec = np.zeros(34, dtype=np.int8)
    for tile in player.hand:
        hand_vec[tile] += 1

    meld_vec = np.zeros(34, dtype=np.int8)
    for meld in player.melds:
        tiles = meld.tiles if meld.type != "kong" else meld.tiles[:3]
        for tile in tiles:
            meld_vec[tile] += 1

    discard_vec = np.zeros(34, dtype=np.int8)
    for seat in game.players:
        for tile in seat.discards:
            discard_vec[tile] += 1

    if override_last_discard is not None:
        last_discard = override_last_discard
    else:
        last_discard = game.discard_pile[-1] if game.discard_pile else -1

    return {
        "hand": hand_vec,
        "melds": meld_vec,
        "discards": discard_vec,
        "tiles_remaining": np.int16(game.tiles_remaining()),
        "current_player": np.int8(game.current_player),
        "last_discard": np.int16(last_discard),
        "stage": np.int8(stage_code),
        "must_discard": np.array([1 if player.must_discard else 0], dtype=np.int8),
    }


def encode_observation(obs: dict) -> np.ndarray:
    """Flatten the observation dictionary into a feature vector."""

    hand = obs["hand"].astype(np.float32) / 4.0
    melds = obs["melds"].astype(np.float32) / 4.0
    discards = obs["discards"].astype(np.float32) / 4.0
    tiles_remaining = np.array([float(obs["tiles_remaining"]) / 136.0], dtype=np.float32)

    current_player = np.zeros(4, dtype=np.float32)
    current_player[int(obs["current_player"])] = 1.0

    last_discard = np.zeros(35, dtype=np.float32)
    last = int(obs["last_discard"])
    if last >= 0:
        last_discard[last] = 1.0
    else:
        last_discard[-1] = 1.0

    stage_vec = np.zeros(4, dtype=np.float32)
    stage_vec[int(obs["stage"])] = 1.0

    must_discard = obs["must_discard"].astype(np.float32)

    return np.concatenate(
        [hand, melds, discards, tiles_remaining, current_player, last_discard, stage_vec, must_discard]
    )


def action_features(action: Tuple[str, Optional[int]]) -> np.ndarray:
    name, tile = action
    type_vec = np.zeros(len(ACTION_TYPES), dtype=np.float32)
    type_vec[ACTION_TYPE_INDEX.get(name, 0)] = 1.0

    tile_vec = np.zeros(35, dtype=np.float32)
    tile_index = 34 if tile is None or tile < 0 else int(tile)
    tile_vec[tile_index] = 1.0

    return np.concatenate([type_vec, tile_vec])


class SharedPolicy:
    """Single softmax policy shared across all four seats."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        learning_rate: float,
        temperature: float,
        baseline_momentum: float,
        reward_scale: float,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.temperature = max(temperature, 1e-6)
        self.baseline_momentum = baseline_momentum
        self.reward_scale = reward_scale
        self.theta = np.zeros(obs_dim * action_dim, dtype=np.float32)
        self.baseline = np.zeros(4, dtype=np.float32)
        self.episode_records: List[dict] = []

    def start_episode(self) -> None:
        self.episode_records.clear()

    def select_action(
        self,
        obs: dict,
        legal_actions: Sequence[Tuple[str, Optional[int]]],
        *,
        seat: int,
        temperature: Optional[float] = None,
    ) -> Tuple[int, Tuple[str, Optional[int]]]:
        obs_vec = encode_observation(obs)
        action_vecs = [action_features(action) for action in legal_actions]
        phi_vectors = [np.kron(vec, obs_vec) for vec in action_vecs]

        logits = np.array([self.theta.dot(phi) for phi in phi_vectors], dtype=np.float32)
        temp = max(temperature or self.temperature, 1e-6)
        scaled = logits / temp
        scaled -= float(np.max(scaled))
        probs = np.exp(scaled)
        probs_sum = float(np.sum(probs))
        if probs_sum <= 0 or not math.isfinite(probs_sum):
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum

        choice = int(np.random.choice(len(legal_actions), p=probs))
        expected_phi = np.zeros_like(self.theta)
        for weight, phi in zip(probs, phi_vectors):
            expected_phi += weight * phi
        self.episode_records.append(
            {
                "seat": seat,
                "phi": phi_vectors[choice],
                "expected": expected_phi,
            }
        )
        return choice, legal_actions[choice]

    def finish_episode(self, rewards: Sequence[float]) -> None:
        if not self.episode_records:
            return

        gradients = np.zeros_like(self.theta)
        scaled_rewards = np.asarray(rewards, dtype=np.float32) / self.reward_scale

        for record in self.episode_records:
            seat = record["seat"]
            advantage = scaled_rewards[seat] - self.baseline[seat]
            gradients += advantage * (record["phi"] - record["expected"])
            if self.baseline_momentum > 0.0:
                self.baseline[seat] += self.baseline_momentum * advantage

        self.theta += self.learning_rate * gradients
        self.episode_records.clear()


class PolicyController:
    """Adapter that lets ``MahjongGame`` query the shared policy."""

    def __init__(self, policy: SharedPolicy, seat: int) -> None:
        self.policy = policy
        self.seat = seat

    def choose_self_action(
        self,
        game,
        player_index: int,
        actions: Sequence[Tuple[str, Optional[int]]],
        _drawn_tile: Optional[int],
    ) -> Optional[Tuple[str, Optional[int]]]:
        legal = [("skip", None)] + list(actions)
        obs = build_observation(game, player_index, STAGE_SELF_ACTION)
        choice, action = self.policy.select_action(obs, legal, seat=self.seat)
        if action[0] == "skip":
            return None
        return action

    def choose_discard(self, game, player_index: int) -> Optional[int]:
        player = game.players[player_index]
        legal = [("discard", tile) for tile in player.hand]
        obs = build_observation(game, player_index, STAGE_DISCARD)
        _, action = self.policy.select_action(obs, legal, seat=self.seat)
        return action[1]

    def choose_reaction(
        self,
        game,
        player_index: int,
        options: Sequence[Tuple[str, Optional[int]]],
        _discarder: int,
        tile: Optional[int],
    ) -> Optional[Tuple[str, Optional[int]]]:
        legal = [("skip", None)] + list(options)
        obs = build_observation(
            game,
            player_index,
            STAGE_REACTION,
            override_last_discard=tile if tile is not None else -1,
        )
        _, action = self.policy.select_action(obs, legal, seat=self.seat)
        if action[0] == "skip":
            return None
        return action


def make_controllers(policy: SharedPolicy) -> List[Optional[PolicyController]]:
    controllers: List[Optional[PolicyController]] = [None] * 4
    for seat in range(1, 4):
        controllers[seat] = PolicyController(policy, seat)
    return controllers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500, help="Number of training rounds")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Policy gradient learning rate")
    parser.add_argument("--temperature", type=float, default=0.3, help="Softmax temperature")
    parser.add_argument(
        "--baseline-momentum",
        type=float,
        default=0.05,
        help="Exponential moving average factor for the per-seat baseline",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=100.0,
        help=(
            "Divisor applied to Mahjong score deltas before updating the policy. "
            "Lower values emphasise the downside of dealing into another player's ron (出衝) "
            "relative to banking a small ready hand, while larger values make the agent "
            "more willing to chase extra fan before declaring (叫糊)."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Window size for the moving averages in the plot",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print Mahjong log output")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where the learning curve plot will be saved",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="hk_mahjong_learning_curve.png",
        help="Filename for the generated learning curve plot",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="How often to print intermediate training statistics",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)

    dummy_obs = {
        "hand": np.zeros(34, dtype=np.int8),
        "melds": np.zeros(34, dtype=np.int8),
        "discards": np.zeros(34, dtype=np.int8),
        "tiles_remaining": np.int16(136),
        "current_player": np.int8(0),
        "last_discard": np.int16(-1),
        "stage": np.int8(0),
        "must_discard": np.array([0], dtype=np.int8),
    }
    obs_dim = encode_observation(dummy_obs).shape[0]
    action_dim = action_features(("discard", 0)).shape[0]

    policy = SharedPolicy(
        obs_dim,
        action_dim,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        baseline_momentum=args.baseline_momentum,
        reward_scale=args.reward_scale,
    )
    controllers = make_controllers(policy)
    env = HongKongMahjongEnv(
        seed=args.seed,
        verbose=args.verbose,
        controllers=controllers,
        agent_index=0,
    )

    reward_window: deque[float] = deque(maxlen=args.window)
    win_window: deque[float] = deque(maxlen=args.window)
    reward_curve: List[float] = []
    win_curve: List[float] = []

    for episode in range(1, args.episodes + 1):
        policy.start_episode()
        obs, info = env.reset()
        done = False

        while not done:
            legal_actions = info.get("legal_actions", [])
            if not legal_actions:
                raise RuntimeError("No legal actions available for the learning seat")
            action_index, action = policy.select_action(obs, legal_actions, seat=0)
            obs, reward, terminated, truncated, info = env.step(action_index)
            done = terminated or truncated

        # Each seat's score already reflects the trade-off between folding and
        # pressing for a larger hand: dealing in (出衝) leaves a large negative
        # delta, while settling for a small ready hand (叫糊) locks in a modest
        # but safe gain.  We feed the raw deltas into the policy update so the
        # gradient step respects that balance.
        final_scores = [float(player.score) for player in env.game.players]
        policy.finish_episode(final_scores)

        scaled_reward = final_scores[0] / args.reward_scale
        reward_window.append(scaled_reward)
        reward_curve.append(float(np.mean(reward_window)))

        winner = None
        if info.get("result"):
            winner = info["result"].get("winner")
        win_window.append(1.0 if winner == 0 else 0.0)
        win_curve.append(float(np.mean(win_window)))

        if episode % args.log_every == 0 or episode == 1:
            print(
                f"Episode {episode:5d} | Seat 1 scaled score: {scaled_reward:+.3f} | "
                f"Avg score: {reward_curve[-1]:+.3f} | Win rate: {win_curve[-1]:.3f}"
            )

    env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / args.plot_name
    episodes = np.arange(1, args.episodes + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(episodes, reward_curve, color="tab:blue", label="Seat 1 mean scaled score")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(f"Mean score / {args.reward_scale:.0f}", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(episodes, win_curve, color="tab:orange", label="Seat 1 win rate")
    ax2.set_ylabel(f"Win rate (window={args.window})", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved learning curve to {plot_path}")
    return plot_path


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

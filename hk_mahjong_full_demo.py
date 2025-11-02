#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Console based Hong Kong Mahjong demo with four players.

This module implements a minimal yet fairly complete Hong Kong Mahjong round
that can be played entirely in the terminal.  It includes the core actions that
were previously missing from the simplified prototype: chi, pon, kong and
winning (tsumo / ron) complete with a scoring model based on common Hong Kong
rules (3 fan minimum, 13 fan limit, doubling system, dealer bonuses).

The implementation focuses on clarity over raw performance so that it can be
used as a learning tool or as the starting point for reinforcement learning
experiments.  The following concessions keep the code compact while still
capturing the requested behaviour:

* No flower tiles or dead wall.  The rule specification discussed earlier uses
  flowers, but handling replacement draws would lengthen the code considerably
  without changing the essential self-play mechanics.  The scoring hooks are in
  place so that flowers can be added later.
* No reach, riichi sticks or kans after reach – these are Japanese mahjong
  concepts and not required here.
* Multiple ron on the same discard are resolved by the first player (closest to
  the discarder).  Extending to multi-ron requires distributing the payment to
  several winners which is outside the immediate scope of this console demo.

Despite these simplifications the demo still covers:

* Shuffling, dealing and drawing from the wall
* Tracking all four players' hands, melds and discards
* Human interaction for player 1, heuristic AI for the other three seats
* Priority resolution between chi/pon/kong/ron after every discard
* Concealed kongs, open kongs, added kongs (upgrading an exposed pon)
* Win hand validation (standard hands, seven pairs, thirteen orphans)
* Fan counting for common Hong Kong hands, including major limit hands
* Payment calculation using the 2^fan "U" system with dealer bonuses

The module exposes a :func:`main` entry-point so it can be executed directly.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    try:
        import gym
        from gym import spaces
    except ImportError:  # pragma: no cover - optional dependency
        gym = None
        spaces = None


GymEnvBase = getattr(gym, "Env", object) if gym is not None else object


# ---------------------------------------------------------------------------
# Tile utilities
# ---------------------------------------------------------------------------

# Tile IDs follow the classic 0-33 encoding:
#   0-8   : 1-9 Man (characters)
#   9-17  : 1-9 Pin (dots)
#   18-26 : 1-9 Sou (bamboo)
#   27-30 : East, South, West, North
#   31-33 : Red, Green, White dragons

SUIT_OFFSETS = {"m": 0, "p": 9, "s": 18}
WINDS = {27: "E", 28: "S", 29: "W", 30: "N"}
DRAGONS = {31: "Red", 32: "Green", 33: "White"}
ALL_TILES = list(range(34))


def tile_name(tile: int) -> str:
    """Return a human readable name for a tile id."""

    if 0 <= tile <= 8:
        return f"{tile + 1}m"
    if 9 <= tile <= 17:
        return f"{tile - 8}p"
    if 18 <= tile <= 26:
        return f"{tile - 17}s"
    if tile in WINDS:
        return WINDS[tile]
    if tile in DRAGONS:
        return DRAGONS[tile][0]
    return f"T{tile}"


def sort_hand(hand: Sequence[int]) -> List[int]:
    """Return a sorted copy of the given tile sequence."""

    return sorted(hand)


def tiles_to_text(hand: Sequence[int]) -> str:
    """Return a compact representation of a hand."""

    return " ".join(tile_name(t) for t in sort_hand(hand))


# ---------------------------------------------------------------------------
# Player data model
# ---------------------------------------------------------------------------


@dataclass
class Meld:
    """Represent a meld (chi, pon or kong)."""

    type: str  # "chi", "pon", "kong"
    tiles: List[int]
    open: bool
    from_player: Optional[int] = None


@dataclass
class PlayerState:
    """Mutable state for a player during a round."""

    name: str
    seat_wind: int
    hand: List[int] = field(default_factory=list)
    melds: List[Meld] = field(default_factory=list)
    discards: List[int] = field(default_factory=list)
    score: int = 0
    must_discard: bool = False

    def sorted_hand(self) -> List[int]:
        return sort_hand(self.hand)

    def remove_tiles(self, tiles: Iterable[int]) -> None:
        for t in tiles:
            self.hand.remove(t)

    def add_meld(self, meld: Meld) -> None:
        self.melds.append(meld)


# ---------------------------------------------------------------------------
# Hand analysis helpers
# ---------------------------------------------------------------------------


def is_seven_pairs(concealed: Sequence[int], melds: Sequence[Meld]) -> bool:
    if melds:
        return False
    if len(concealed) != 14:
        return False
    counter = Counter(concealed)
    return all(count == 2 for count in counter.values())


def is_thirteen_orphans(concealed: Sequence[int], melds: Sequence[Meld]) -> bool:
    if melds:
        return False
    needed = {
        0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33,
    }
    counter = Counter(concealed)
    if not needed.issubset(counter.keys()):
        return False
    if len(concealed) != 14:
        return False
    pair_found = False
    for tile in needed:
        if counter[tile] >= 2:
            pair_found = True
            break
    return pair_found and sum(counter.values()) == 14


def _search_standard(counter: Counter, pair: Optional[int], sets: List[Tuple[str, Tuple[int, ...]]]) -> Optional[Tuple[int, List[Tuple[str, Tuple[int, ...]]]]]:
    remaining = sum(counter.values())
    if remaining == 0:
        if pair is not None:
            return pair, sets.copy()
        return None

    tile = min(t for t, cnt in counter.items() if cnt > 0)

    # Try to take a pair
    if pair is None and counter[tile] >= 2:
        counter[tile] -= 2
        if counter[tile] == 0:
            del counter[tile]
        result = _search_standard(counter, tile, sets)
        counter[tile] = counter.get(tile, 0) + 2
        if result:
            return result

    # Try to take a pung
    if counter[tile] >= 3:
        counter[tile] -= 3
        if counter[tile] == 0:
            del counter[tile]
        sets.append(("pon", (tile, tile, tile)))
        result = _search_standard(counter, pair, sets)
        sets.pop()
        counter[tile] = counter.get(tile, 0) + 3
        if result:
            return result

    # Try to take a chow
    suit = tile // 9
    pos_in_suit = tile % 9
    if tile <= 26 and pos_in_suit <= 6:
        t1, t2 = tile + 1, tile + 2
        if counter.get(t1, 0) > 0 and counter.get(t2, 0) > 0 and t1 // 9 == suit and t2 // 9 == suit:
            counter[tile] -= 1
            counter[t1] -= 1
            counter[t2] -= 1
            for t in (tile, t1, t2):
                if counter[t] == 0:
                    del counter[t]
            sets.append(("chi", (tile, t1, t2)))
            result = _search_standard(counter, pair, sets)
            sets.pop()
            for t in (tile, t1, t2):
                counter[t] = counter.get(t, 0) + 1
            if result:
                return result

    return None


def decompose_standard(concealed: Sequence[int]) -> Optional[Tuple[int, List[Tuple[str, Tuple[int, ...]]]]]:
    counter = Counter(concealed)
    return _search_standard(counter, None, [])


def is_all_triplets(melds: Sequence[Meld], sets: Sequence[Tuple[str, Tuple[int, ...]]]) -> bool:
    for m in melds:
        if m.type == "chi":
            return False
    for set_type, _tiles in sets:
        if set_type != "pon":
            return False
    return True


def determine_suits(all_tiles: Sequence[int]) -> Tuple[set, bool]:
    suits = set()
    has_honor = False
    for t in all_tiles:
        if 0 <= t <= 26:
            suits.add(t // 9)
        else:
            has_honor = True
    return suits, has_honor


def count_kongs(melds: Sequence[Meld]) -> Tuple[int, int]:
    """Return (open_kongs, concealed_kongs)."""

    open_kong = concealed_kong = 0
    for m in melds:
        if m.type == "kong":
            if m.open:
                open_kong += 1
            else:
                concealed_kong += 1
    return open_kong, concealed_kong


def fan_from_hand(
    player: PlayerState,
    concealed: Sequence[int],
    melds: Sequence[Meld],
    win_tile: int,
    win_type: str,
    round_wind: int,
) -> Tuple[int, List[Tuple[str, int]]]:
    """Compute the fan value and breakdown for a winning hand.

    Returns ``(fan_total, [(name, fan_value), ...])``.  The fan total is capped
    at 13.  The caller must ensure the hand is a valid mahjong hand prior to
    invoking this function.
    """

    breakdown: List[Tuple[str, int]] = []
    fan_total = 0

    total_tiles = list(concealed)
    pair_tile: Optional[int] = None
    standard_sets: List[Tuple[str, Tuple[int, ...]]] = []

    if is_thirteen_orphans(concealed, melds):
        breakdown.append(("Thirteen Orphans", 13))
        return 13, breakdown

    if is_seven_pairs(concealed, melds):
        breakdown.append(("Seven Pairs", 4))
        fan_total += 4
        pair_tile = None  # Not used later
    else:
        result = decompose_standard(concealed)
        if result is None:
            # Should not happen if caller validated the win
            return 0, []
        pair_tile, standard_sets = result

    if win_type == "tsumo":
        breakdown.append(("Self Draw", 1))
        fan_total += 1

    # Pungs / kongs detection
    pung_tiles: List[int] = []
    chow_present = False
    for meld in melds:
        if meld.type == "chi":
            chow_present = True
        elif meld.type in {"pon", "kong"}:
            pung_tiles.append(meld.tiles[0])
    for set_type, tiles in standard_sets:
        if set_type == "chi":
            chow_present = True
        else:
            pung_tiles.append(tiles[0])

    # Dragon/seat/round winds
    for tile in pung_tiles:
        if tile in DRAGONS:
            breakdown.append((f"Dragon {tile_name(tile)}", 1))
            fan_total += 1
        if tile == player.seat_wind:
            breakdown.append(("Seat Wind", 1))
            fan_total += 1
        if tile == round_wind:
            breakdown.append(("Round Wind", 1))
            fan_total += 1

    if not chow_present:
        breakdown.append(("All Pungs", 3))
        fan_total += 3

    # Suit based hands
    all_tiles = list(concealed)
    for meld in melds:
        all_tiles.extend(meld.tiles if meld.type != "kong" else meld.tiles[:3])

    suits, has_honor = determine_suits(all_tiles)
    if len(suits) == 1:
        if has_honor:
            breakdown.append(("Half Flush", 3))
            fan_total += 3
        else:
            breakdown.append(("Pure One Suit", 6))
            fan_total += 6

    # Major limit hands (override other scoring but still recorded for clarity)
    dragon_counts = {tile: 0 for tile in DRAGONS}
    wind_counts = {tile: 0 for tile in WINDS}
    for tile in pung_tiles:
        if tile in dragon_counts:
            dragon_counts[tile] += 1
        if tile in wind_counts:
            wind_counts[tile] += 1

    if all(count >= 1 for count in dragon_counts.values()):
        breakdown.append(("Big Three Dragons", 13))
        return 13, breakdown

    wind_pungs = sum(1 for count in wind_counts.values() if count >= 1)
    if wind_pungs == 4:
        breakdown.append(("Big Four Winds", 13))
        return 13, breakdown
    if wind_pungs == 3 and pair_tile in WINDS:
        breakdown.append(("Little Four Winds", 13))
        return 13, breakdown

    if not suits and has_honor:
        breakdown.append(("All Honors", 13))
        return 13, breakdown

    # Kong bonus
    open_kong, concealed_kong = count_kongs(melds)
    if open_kong:
        breakdown.append(("Open Kong", open_kong))
        fan_total += open_kong
    if concealed_kong:
        breakdown.append(("Concealed Kong", 2 * concealed_kong))
        fan_total += 2 * concealed_kong

    return min(fan_total, 13), breakdown


def can_win_hand(concealed: Sequence[int], melds: Sequence[Meld]) -> bool:
    return (
        is_thirteen_orphans(concealed, melds)
        or is_seven_pairs(concealed, melds)
        or decompose_standard(concealed) is not None
    )


# ---------------------------------------------------------------------------
# Game engine
# ---------------------------------------------------------------------------


class MahjongGame:
    """Single-round Hong Kong Mahjong game loop."""

    def __init__(
        self,
        *,
        rng: Optional[random.Random] = None,
        verbose: bool = True,
        controllers: Optional[Sequence[object]] = None,
    ) -> None:
        self.round_wind = 27  # East
        self.dealer = 0
        self.players = [
            PlayerState("You", 27),
            PlayerState("AI-2", 28),
            PlayerState("AI-3", 29),
            PlayerState("AI-4", 30),
        ]
        self.wall: List[int] = []
        self.discard_pile: List[int] = []
        self.step_counter = 1
        self.current_player = self.dealer
        self.verbose = verbose
        self.rng = rng or random.Random()
        self.round_over = False
        self.result: Optional[dict] = None
        self.last_drawn_tile: Optional[int] = None
        self.controllers: Dict[int, object] = {}
        if controllers:
            for idx, controller in enumerate(controllers):
                if controller is not None:
                    self.controllers[idx] = controller

    # ----- Utility logging -------------------------------------------------

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[{self.step_counter:03d}] {message}")
        self.step_counter += 1

    # ----- Initial setup ---------------------------------------------------

    def build_wall(self) -> None:
        self.wall = []
        for tile in ALL_TILES:
            self.wall.extend([tile] * 4)
        self.rng.shuffle(self.wall)
        self.discard_pile.clear()

    def deal_tiles(self) -> None:
        self.round_over = False
        self.result = None
        self.last_drawn_tile = None
        for player in self.players:
            player.hand.clear()
            player.melds.clear()
            player.discards.clear()
            player.must_discard = False

        # Deal 13 tiles to everyone
        for _ in range(13):
            for player in self.players:
                player.hand.append(self.wall.pop())

        for player in self.players:
            player.hand.sort()

        # Dealer draws an extra tile to start
        dealer = self.players[self.dealer]
        dealer.hand.append(self.wall.pop())
        dealer.must_discard = True
        dealer.hand.sort()

        self.current_player = self.dealer
        self.step_counter = 1
        self.log("Tiles shuffled and dealt. Dealer draws an extra tile.")
        self.show_game_state()

    # ----- Helper queries --------------------------------------------------

    def tiles_remaining(self) -> int:
        return len(self.wall)

    def discards_count(self) -> int:
        return len(self.discard_pile)

    def draw_tile(self, player: PlayerState) -> Optional[int]:
        if not self.wall:
            return None
        tile = self.wall.pop()
        player.hand.append(tile)
        player.hand.sort()
        self.last_drawn_tile = tile
        return tile

    # ----- Action discovery ------------------------------------------------

    def available_concealed_kongs(self, player: PlayerState) -> List[int]:
        counter = Counter(player.hand)
        return [tile for tile, cnt in counter.items() if cnt == 4]

    def available_added_kongs(self, player: PlayerState) -> List[int]:
        tiles: List[int] = []
        for meld in player.melds:
            if meld.type == "pon":
                tile = meld.tiles[0]
                if player.hand.count(tile) >= 1:
                    tiles.append(tile)
        return tiles

    def available_self_actions(
        self,
        player_index: int,
        drawn_tile: int,
    ) -> List[Tuple[str, Optional[int]]]:
        player = self.players[player_index]
        actions: List[Tuple[str, Optional[int]]] = []

        if can_win_hand(player.hand, player.melds):
            actions.append(("tsumo", drawn_tile))

        for tile in self.available_concealed_kongs(player):
            actions.append(("concealed_kong", tile))

        for tile in self.available_added_kongs(player):
            actions.append(("added_kong", tile))

        return actions

    def reactions_to_discard(
        self, discarder: int, tile: int
    ) -> List[Tuple[int, Tuple[str, Optional[int]]]]:
        responses: List[Tuple[int, Tuple[str, Optional[int]]]] = []
        for offset in range(1, 4):
            idx = (discarder + offset) % 4
            player = self.players[idx]
            candidate_hand = player.hand + [tile]
            if can_win_hand(candidate_hand, player.melds):
                responses.append((idx, ("ron", tile)))
                continue

            counter = Counter(player.hand)
            if counter[tile] >= 3:
                responses.append((idx, ("kong", tile)))
                continue
            if counter[tile] >= 2:
                responses.append((idx, ("pon", tile)))

            # Chi only for next player
            if offset == 1 and tile <= 26:
                suit = tile // 9
                value = tile % 9
                chi_options = []
                if value >= 2:
                    chi = [tile - 2, tile - 1, tile]
                    if all(player.hand.count(t) >= (1 if t != tile else 0) for t in chi if t != tile) and all(
                        t // 9 == suit for t in chi
                    ):
                        chi_options.append(chi)
                if value >= 1 and value <= 7:
                    chi = [tile - 1, tile, tile + 1]
                    if all(player.hand.count(t) >= (1 if t != tile else 0) for t in chi if t != tile) and all(
                        t // 9 == suit for t in chi
                    ):
                        chi_options.append(chi)
                if value <= 6:
                    chi = [tile, tile + 1, tile + 2]
                    if all(player.hand.count(t) >= (1 if t != tile else 0) for t in chi if t != tile) and all(
                        t // 9 == suit for t in chi
                    ):
                        chi_options.append(chi)
                if chi_options:
                    responses.append((idx, ("chi", chi_options[0][0])))
        return responses

    # ----- Game state display ---------------------------------------------

    def show_game_state(self) -> None:
        print("\nCurrent state:")
        for idx, player in enumerate(self.players):
            hand = tiles_to_text(player.hand) if idx == 0 else f"{len(player.hand)} tiles"
            melds = [f"{m.type}:{tiles_to_text(m.tiles)}" for m in player.melds]
            print(
                f"  Player {idx+1} ({player.name}) - Hand: {hand} | Melds: {', '.join(melds) if melds else 'None'} | Discards: {tiles_to_text(player.discards)}"
            )
        print(
            f"  Tiles remaining in wall: {self.tiles_remaining()} | Total discards: {self.discards_count()}\n"
        )

    # ----- Action execution ------------------------------------------------

    def execute_concealed_kong(self, player: PlayerState, tile: int) -> Optional[int]:
        player.remove_tiles([tile] * 4)
        player.add_meld(Meld("kong", [tile] * 4, open=False))
        self.log(f"{player.name} declares concealed kong of {tile_name(tile)}")
        supplement = self.draw_tile(player)
        if supplement is None:
            self.log("Wall is empty after concealed kong.")
        else:
            self.log(f"{player.name} draws supplement tile {tile_name(supplement)}")
        return supplement

    def execute_added_kong(self, player: PlayerState, tile: int) -> Optional[int]:
        for meld in player.melds:
            if meld.type == "pon" and meld.tiles[0] == tile:
                meld.type = "kong"
                meld.tiles.append(tile)
                meld.open = True
                break
        player.remove_tiles([tile])
        self.log(f"{player.name} upgrades pon to kong with {tile_name(tile)}")
        supplement = self.draw_tile(player)
        if supplement is None:
            self.log("Wall is empty after added kong.")
        else:
            self.log(f"{player.name} draws supplement tile {tile_name(supplement)}")
        return supplement

    def execute_open_kong(self, player_index: int, tile: int, from_player: int) -> Optional[int]:
        player = self.players[player_index]
        player.remove_tiles([tile] * 3)
        player.add_meld(Meld("kong", [tile] * 4, open=True, from_player=from_player))
        self.log(
            f"{player.name} declares open kong on {tile_name(tile)} from Player {from_player+1}"
        )
        supplement = self.draw_tile(player)
        if supplement is None:
            self.log("Wall is empty after kong supplement draw.")
        else:
            self.log(f"{player.name} draws supplement tile {tile_name(supplement)}")
        return supplement

    def execute_pon(
        self, player_index: int, tile: Optional[int], from_player: int
    ) -> bool:
        player = self.players[player_index]
        if tile is None:
            self.log(
                f"{player.name} attempted pon but no tile was provided; treating as skip."
            )
            return False
        if player.hand.count(tile) < 2:
            tile_label = tile_name(tile) if isinstance(tile, int) else str(tile)
            self.log(
                f"{player.name} attempted pon on {tile_label} but lacked the tiles; "
                "treating as skip."
            )
            return False
        player.remove_tiles([tile] * 2)
        player.add_meld(Meld("pon", [tile] * 3, open=True, from_player=from_player))
        self.log(
            f"{player.name} calls pon on {tile_name(tile)} from Player {from_player+1}"
        )
        return True

    def execute_chi(
        self, player_index: int, base_tile: int, from_player: int, discard_tile: int
    ) -> None:
        player = self.players[player_index]

        if discard_tile > 26:
            raise ValueError("Chi is only available on suited tiles")

        suit = discard_tile // 9
        candidates: List[Tuple[int, List[int], List[int]]] = []
        for start in range(discard_tile - 2, discard_tile + 1):
            if start < 0 or start > 26:
                continue
            if start // 9 != suit:
                continue
            sequence = [start, start + 1, start + 2]
            if discard_tile not in sequence:
                continue
            needed = [t for t in sequence if t != discard_tile]
            if all(player.hand.count(t) >= needed.count(t) for t in set(needed)):
                candidates.append((start, sequence, needed))

        if not candidates:
            raise ValueError("Player cannot form chi with the provided discard")

        chosen_sequence: Optional[Tuple[List[int], List[int]]] = None
        for start, sequence, needed in candidates:
            if start == base_tile:
                chosen_sequence = (sequence, needed)
                break

        if chosen_sequence is None:
            sequence, needed = candidates[0][1], candidates[0][2]
        else:
            sequence, needed = chosen_sequence

        player.remove_tiles(needed)

        meld_tiles = sorted(sequence)
        player.add_meld(Meld("chi", meld_tiles, open=True, from_player=from_player))
        self.log(
            f"{player.name} calls chi {tiles_to_text(meld_tiles)} from Player {from_player+1}"
        )

    # ----- AI heuristics ---------------------------------------------------

    def set_controller(self, player_index: int, controller: Optional[object]) -> None:
        """Attach or remove a controller for the given player."""

        if controller is None:
            self.controllers.pop(player_index, None)
        else:
            self.controllers[player_index] = controller

    def ai_choose_self_action(
        self,
        actions: List[Tuple[str, Optional[int]]],
        player_index: Optional[int] = None,
        drawn_tile: Optional[int] = None,
    ) -> Optional[Tuple[str, Optional[int]]]:
        if not actions:
            return None
        if player_index is None:
            player_index = self.current_player
        controller = self.controllers.get(player_index)
        if controller and hasattr(controller, "choose_self_action"):
            choice = controller.choose_self_action(self, player_index, actions, drawn_tile)
            if choice is not None:
                return choice
        priority = {"tsumo": 3, "concealed_kong": 2, "added_kong": 2}
        actions.sort(key=lambda a: priority.get(a[0], 0), reverse=True)
        return actions[0]

    def ai_discard(
        self, player: PlayerState, player_index: Optional[int] = None
    ) -> Optional[int]:
        """Return the tile the AI elects to discard, or ``None`` if impossible."""

        if player_index is None:
            player_index = self.players.index(player)

        if not player.hand:
            self.log(
                f"{player.name} has no tiles available to discard; passing the turn."
            )
            return None

        controller = self.controllers.get(player_index)
        if controller and hasattr(controller, "choose_discard"):
            tile_choice = controller.choose_discard(self, player_index)
            if tile_choice is not None and tile_choice in player.hand:
                player.hand.remove(tile_choice)
                return tile_choice

        if not player.hand:
            self.log(
                f"{player.name} still has no tiles to discard after policy selection; "
                "passing the turn."
            )
            return None

        tile = max(player.hand)
        player.hand.remove(tile)
        return tile

    def ai_reaction(
        self, player_index: int, options: List[Tuple[str, Optional[int]]]
    ) -> Optional[Tuple[str, Optional[int]]]:
        if not options:
            return None
        priority = {"ron": 3, "kong": 2, "pon": 1, "chi": 0}
        options.sort(key=lambda a: priority.get(a[0], -1), reverse=True)
        return options[0]

    # ----- Winning and scoring --------------------------------------------

    def resolve_win(
        self,
        winner_index: int,
        win_type: str,
        tile: int,
        discarder: Optional[int] = None,
    ) -> None:
        player = self.players[winner_index]
        concealed = player.sorted_hand()
        fan_total, breakdown = fan_from_hand(player, concealed, player.melds, tile, win_type, self.round_wind)

        if fan_total < 3:
            self.log(
                f"{player.name}'s hand only has {fan_total} fan (<3). Win rejected."
            )
            # Remove the tile that was temporarily added for evaluation
            if win_type == "ron" and discarder is not None:
                player.hand.remove(tile)
            return

        self.log(
            f"{player.name} wins by {'Tsumo' if win_type == 'tsumo' else 'Ron'} with {fan_total} fan."
        )
        for name, fan in breakdown:
            self.log(f"  - {name}: {fan} fan")

        effective_fan = min(fan_total, 13)
        unit = 2 ** effective_fan

        if win_type == "tsumo":
            total_gain = 0
            for idx, opponent in enumerate(self.players):
                if idx == winner_index:
                    continue
                payment = 2 * unit if idx == self.dealer else unit
                if winner_index == self.dealer:
                    payment = 2 * unit
                opponent.score -= payment
                total_gain += payment
            player.score += total_gain
        else:  # ron
            assert discarder is not None
            payment = 6 * unit if winner_index == self.dealer else 4 * unit
            self.players[discarder].score -= payment
            player.score += payment

        self.show_game_state()
        self.log("Round finished.")
        self.round_over = True
        self.result = {
            "winner": winner_index,
            "win_type": win_type,
            "tile": tile,
            "fan": fan_total,
            "breakdown": breakdown,
            "discarder": discarder,
        }

    # ----- Main loop -------------------------------------------------------

    def play_round(self) -> None:
        self.build_wall()
        self.deal_tiles()

        while True:
            player = self.players[self.current_player]

            if not player.must_discard:
                tile = self.draw_tile(player)
                if tile is None:
                    self.log("The wall is empty. Round ends in a draw.")
                    break
                self.log(
                    f"Player {self.current_player+1} ({player.name}) draws {tile_name(tile)}"
                )

                actions = self.available_self_actions(self.current_player, tile)
                if self.current_player == 0:
                    self.prompt_self_actions(actions)
                else:
                    choice = self.ai_choose_self_action(
                        actions, self.current_player, tile
                    )
                    if choice:
                        self.process_self_action(self.current_player, choice)
                        if self.round_over:
                            return
                        continue

                player.must_discard = True

            discard_tile = self.prompt_discard(self.current_player)
            self.discard_pile.append(discard_tile)
            player.discards.append(discard_tile)
            self.log(
                f"Player {self.current_player+1} ({player.name}) discards {tile_name(discard_tile)}"
            )
            player.must_discard = False

            reactions = self.reactions_to_discard(self.current_player, discard_tile)
            chosen_reaction = self.resolve_reactions(
                self.current_player, reactions, discard_tile
            )

            if chosen_reaction is None:
                self.current_player = (self.current_player + 1) % 4
                continue

            actor_idx, action = chosen_reaction
            actor = self.players[actor_idx]

            if action[0] == "ron":
                actor.hand.append(discard_tile)
                actor.hand.sort()
                self.resolve_win(actor_idx, "ron", discard_tile, discarder=self.current_player)
                return
            elif action[0] == "kong":
                self.execute_open_kong(actor_idx, discard_tile, self.current_player)
                actor.must_discard = True
                self.current_player = actor_idx
            elif action[0] == "pon":
                if self.execute_pon(actor_idx, discard_tile, self.current_player):
                    actor.must_discard = True
                    self.current_player = actor_idx
                else:
                    self.current_player = (self.current_player + 1) % 4
            elif action[0] == "chi":
                self.execute_chi(
                    actor_idx,
                    action[1] or discard_tile,
                    self.current_player,
                    discard_tile,
                )
                actor.must_discard = True
                self.current_player = actor_idx

    # ----- Interaction helpers --------------------------------------------

    def prompt_self_actions(self, actions: List[Tuple[str, Optional[int]]]) -> None:
        if not actions:
            return
        print("Available self actions:")
        for idx, (name, tile) in enumerate(actions):
            label = tile_name(tile) if tile is not None else ""
            print(f"  [{idx}] {name} {label}")
        while True:
            choice = input("Choose action or press Enter to skip: ").strip()
            if choice == "":
                return
            if choice.isdigit() and 0 <= int(choice) < len(actions):
                action = actions[int(choice)]
                self.process_self_action(0, action)
                if action[0] == "tsumo" and self.round_over:
                    return
                return
            print("Invalid choice. Try again.")

    def process_self_action(
        self, player_index: int, action: Tuple[str, Optional[int]]
    ) -> Optional[int]:
        player = self.players[player_index]
        name, tile = action
        if name == "tsumo":
            self.resolve_win(player_index, "tsumo", tile or player.hand[-1])
            return tile
        elif name == "concealed_kong" and tile is not None:
            return self.execute_concealed_kong(player, tile)
        elif name == "added_kong" and tile is not None:
            return self.execute_added_kong(player, tile)
        else:
            return None

    def prompt_discard(self, player_index: int) -> int:
        player = self.players[player_index]
        if player_index == 0:
            while True:
                print(f"Your hand: {tiles_to_text(player.hand)}")
                print(
                    f"Tiles remaining: {self.tiles_remaining()} | Discards: {self.discards_count()}"
                )
                choice = input("Select tile index to discard (0-based): ").strip()
                if choice.isdigit():
                    idx = int(choice)
                    if 0 <= idx < len(player.hand):
                        return player.hand.pop(idx)
                print("Invalid input, try again.")
        else:
            tile = self.ai_discard(player, player_index)
            if tile is None:
                raise RuntimeError(
                    f"{player.name} had no tiles to discard during CLI play."
                )
            return tile

    def resolve_reactions(
        self,
        discarder: int,
        reactions: List[Tuple[int, Tuple[str, Optional[int]]]],
        tile: Optional[int] = None,
    ) -> Optional[Tuple[int, Tuple[str, Optional[int]]]]:
        if not reactions:
            return None
        priority = {"ron": 3, "kong": 2, "pon": 1, "chi": 0}
        options_by_player: Dict[int, List[Tuple[str, Optional[int]]]] = {}
        for idx, action in reactions:
            options_by_player.setdefault(idx, []).append(action)

        chosen: List[Tuple[int, Tuple[str, Optional[int]]]] = []
        for idx, options in options_by_player.items():
            controller = self.controllers.get(idx)
            choice: Optional[Tuple[str, Optional[int]]] = None
            if controller and hasattr(controller, "choose_reaction"):
                choice = controller.choose_reaction(
                    self, idx, options, discarder, tile
                )
                if choice is not None and choice not in options:
                    choice = None
            if choice is None and options:
                options.sort(key=lambda a: priority.get(a[0], -1), reverse=True)
                choice = options[0]
            if choice is not None:
                chosen.append((idx, choice))

        if not chosen:
            return None

        def sort_key(item: Tuple[int, Tuple[str, Optional[int]]]) -> Tuple[int, int]:
            idx, action = item
            relative = (idx - discarder) % 4
            return (-priority.get(action[0], -1), relative)

        chosen.sort(key=sort_key)
        idx, action = chosen[0]
        actor = self.players[idx]
        self.log(
            f"Player {idx+1} ({actor.name}) chooses {action[0]}"
        )
        return idx, action


# ---------------------------------------------------------------------------
# OpenAI Gym compatible environment
# ---------------------------------------------------------------------------


class HongKongMahjongEnv(GymEnvBase):
    """OpenAI Gym style environment that wraps :class:`MahjongGame`."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        verbose: bool = False,
        agent_index: int = 0,
        controllers: Optional[Sequence[object]] = None,
    ) -> None:
        if spaces is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "HongKongMahjongEnv requires the 'gymnasium' (preferred) or 'gym' package. "
                "Install one of them with 'pip install gymnasium' or 'pip install gym'."
            )

        self.random = random.Random(seed)
        self.game = MahjongGame(
            rng=self.random, verbose=verbose, controllers=controllers
        )
        if not 0 <= agent_index < 4:
            raise ValueError("agent_index must be between 0 and 3")
        self.agent_index = agent_index

        self.action_space = spaces.Discrete(64)
        self.observation_space = spaces.Dict(
            {
                "hand": spaces.Box(low=0, high=4, shape=(34,), dtype=np.int8),
                "melds": spaces.Box(low=0, high=4, shape=(34,), dtype=np.int8),
                "discards": spaces.Box(low=0, high=16, shape=(34,), dtype=np.int8),
                "tiles_remaining": spaces.Box(low=0, high=136, shape=(), dtype=np.int16),
                "current_player": spaces.Discrete(4),
                "last_discard": spaces.Box(low=-1, high=33, shape=(), dtype=np.int16),
                "stage": spaces.Discrete(4),
                "must_discard": spaces.MultiBinary(1),
            }
        )

        self._legal_actions: List[Tuple[str, Optional[int]]] = []
        self._stage: Optional[str] = None
        self._pending_reactions: List[Tuple[int, Tuple[str, Optional[int]]]] = []
        self._reaction_discarder: Optional[int] = None
        self._reaction_tile: Optional[int] = None
        self._terminal_reason: Optional[str] = None
        self._pending_draw_override: Optional[int] = None

    def set_controller(self, player_index: int, controller: Optional[object]) -> None:
        """Update the controller responsible for a non-learning seat."""

        if player_index == self.agent_index:
            raise ValueError("Cannot attach a controller to the learning agent seat")
        self.game.set_controller(player_index, controller)

    # ----- Gym API ---------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.random.seed(seed)
        self.game.build_wall()
        self.game.deal_tiles()
        self._legal_actions.clear()
        self._stage = None
        self._pending_reactions = []
        self._reaction_discarder = None
        self._reaction_tile = None
        self._terminal_reason = None
        self._pending_draw_override = None
        for player in self.game.players:
            player.score = 0

        self._advance_until_agent_turn()
        return self._make_observation(), {"legal_actions": self._legal_actions.copy()}

    def step(self, action: int):
        if self.game.round_over:
            raise RuntimeError("Cannot call step() when the round is over. Call reset().")
        if action >= len(self._legal_actions):
            raise ValueError(
                f"Action index {action} is invalid for the current stage with "
                f"{len(self._legal_actions)} legal actions."
            )

        chosen = self._legal_actions[action]
        if self._stage == "discard":
            self._apply_discard(chosen)
        elif self._stage == "self_action":
            self._apply_self_action(chosen)
        elif self._stage == "reaction":
            self._apply_reaction(chosen)
        else:
            raise RuntimeError(f"Unexpected stage {self._stage}")

        self._advance_until_agent_turn()

        terminated = bool(self.game.round_over)
        truncated = False
        reward = 0.0

        if terminated:
            if self.game.result and self.game.result.get("winner") == self.agent_index:
                reward = 1.0
            elif self.game.result and self.game.result.get("winner") is not None:
                reward = -1.0

        observation = self._make_observation()
        info = {
            "legal_actions": self._legal_actions.copy(),
            "stage": self._stage,
            "result": self.game.result,
            "terminal_reason": self._terminal_reason,
        }
        return observation, reward, terminated, truncated, info

    # ----- Internal helpers ------------------------------------------------

    def _clear_reaction_context(self) -> None:
        self._pending_reactions = []
        self._reaction_discarder = None
        self._reaction_tile = None
        self._legal_actions = []
        self._stage = None

    def _make_observation(self) -> dict:
        player = self.game.players[self.agent_index]
        hand_vec = np.zeros(34, dtype=np.int8)
        for tile in player.hand:
            hand_vec[tile] += 1

        meld_vec = np.zeros(34, dtype=np.int8)
        for meld in player.melds:
            tiles = meld.tiles if meld.type != "kong" else meld.tiles[:3]
            for tile in tiles:
                meld_vec[tile] += 1

        discard_vec = np.zeros(34, dtype=np.int8)
        for p in self.game.players:
            for tile in p.discards:
                discard_vec[tile] += 1

        stage_mapping = {None: 0, "self_action": 1, "discard": 2, "reaction": 3}
        last_discard = self.game.discard_pile[-1] if self.game.discard_pile else -1

        return {
            "hand": hand_vec,
            "melds": meld_vec,
            "discards": discard_vec,
            "tiles_remaining": np.int16(self.game.tiles_remaining()),
            "current_player": np.int8(self.game.current_player),
            "last_discard": np.int16(last_discard),
            "stage": np.int8(stage_mapping.get(self._stage, 0)),
            "must_discard": np.array(
                [1 if self.game.players[self.agent_index].must_discard else 0], dtype=np.int8
            ),
        }

    def _advance_until_agent_turn(self) -> None:
        while not self.game.round_over:
            player = self.game.players[self.game.current_player]
            if self.game.current_player == self.agent_index:
                if not player.must_discard:
                    if self._pending_draw_override is not None:
                        tile = self._pending_draw_override
                        self._pending_draw_override = None
                    else:
                        tile = self.game.draw_tile(player)
                    if tile is None:
                        self._terminal_reason = "wall_empty"
                        self.game.round_over = True
                        self.game.result = {"winner": None, "win_type": "draw"}
                        return
                    self.game.log(
                        f"Player {self.agent_index+1} ({player.name}) draws {tile_name(tile)}"
                    )
                    actions = self.game.available_self_actions(self.agent_index, tile)
                    if actions:
                        self._stage = "self_action"
                        self._legal_actions = [("skip", None)] + actions
                        return
                    player.must_discard = True
                if player.must_discard:
                    if not player.hand:
                        self.game.log(
                            f"Player {self.agent_index+1} ({player.name}) has no tiles to discard; "
                            "passing the turn."
                        )
                        player.must_discard = False
                        self.game.current_player = (self.game.current_player + 1) % 4
                        continue
                    self._stage = "discard"
                    self._legal_actions = [("discard", tile) for tile in player.hand]
                    return
            else:
                if self._perform_ai_turn():
                    return
            if self.game.round_over:
                return

        self._stage = None
        self._legal_actions = []

    def _perform_ai_turn(self) -> bool:
        idx = self.game.current_player
        player = self.game.players[idx]

        if not player.must_discard:
            tile = self.game.draw_tile(player)
            if tile is None:
                self._terminal_reason = "wall_empty"
                self.game.round_over = True
                self.game.result = {"winner": None, "win_type": "draw"}
                return False
            self.game.log(f"Player {idx+1} ({player.name}) draws {tile_name(tile)}")

            actions = self.game.available_self_actions(idx, tile)
            choice = self.game.ai_choose_self_action(actions, idx, tile)
            if choice:
                self.game.process_self_action(idx, choice)
                if self.game.round_over:
                    return False
                if choice[0] in {"concealed_kong", "added_kong"}:
                    player.must_discard = True
                    return False
            player.must_discard = True

        discard_tile = self.game.ai_discard(player, idx)
        if discard_tile is None:
            self.game.log(
                f"Player {idx+1} ({player.name}) cannot discard; advancing to the next seat."
            )
            player.must_discard = False
            self.game.current_player = (idx + 1) % 4
            return False
        self.game.discard_pile.append(discard_tile)
        player.discards.append(discard_tile)
        self.game.log(
            f"Player {idx+1} ({player.name}) discards {tile_name(discard_tile)}"
        )
        player.must_discard = False

        reactions = self.game.reactions_to_discard(idx, discard_tile)
        if self._prepare_reactions(idx, discard_tile, reactions):
            return True

        self.game.current_player = (idx + 1) % 4
        return False

    def _prepare_reactions(
        self,
        discarder: int,
        tile: int,
        reactions: List[Tuple[int, Tuple[str, Optional[int]]]],
    ) -> bool:
        if not reactions:
            return False

        agent_options = [action for idx, action in reactions if idx == self.agent_index]
        if agent_options:
            self._stage = "reaction"
            self._legal_actions = [("skip", None)] + agent_options
            self._pending_reactions = reactions
            self._reaction_discarder = discarder
            self._reaction_tile = tile
            return True

        chosen = self.game.resolve_reactions(discarder, reactions, tile)
        if chosen is None:
            return False
        actor_idx, action = chosen
        self._clear_reaction_context()
        self._execute_reaction(actor_idx, action, discarder, tile)
        return False

    def _apply_discard(self, action: Tuple[str, Optional[int]]) -> None:
        if action[0] != "discard" or action[1] is None:
            raise ValueError("Invalid discard action")
        player = self.game.players[self.agent_index]
        tile = action[1]
        player.hand.remove(tile)
        self.game.discard_pile.append(tile)
        player.discards.append(tile)
        self.game.log(
            f"Player {self.agent_index+1} ({player.name}) discards {tile_name(tile)}"
        )
        player.must_discard = False

        reactions = self.game.reactions_to_discard(self.agent_index, tile)
        if self._prepare_reactions(self.agent_index, tile, reactions):
            return

        self.game.current_player = (self.agent_index + 1) % 4

    def _apply_self_action(self, action: Tuple[str, Optional[int]]) -> None:
        if action[0] == "skip":
            self.game.players[self.agent_index].must_discard = True
            return
        supplement = self.game.process_self_action(self.agent_index, action)
        if not self.game.round_over and action[0] in {"concealed_kong", "added_kong"}:
            self.game.players[self.agent_index].must_discard = False
            self._pending_draw_override = supplement

    def _apply_reaction(self, action: Tuple[str, Optional[int]]) -> None:
        if self._reaction_discarder is None or self._reaction_tile is None:
            raise RuntimeError("Reaction context is missing")

        if action[0] == "skip":
            discarder = self._reaction_discarder
            tile = self._reaction_tile
            remaining = [item for item in self._pending_reactions if item[0] != self.agent_index]
            if remaining:
                chosen = self.game.resolve_reactions(
                    discarder, remaining, tile
                )
                if chosen:
                    self._clear_reaction_context()
                    actor_idx, act = chosen
                    self._execute_reaction(actor_idx, act, discarder, tile)
                    return
            self.game.current_player = (discarder + 1) % 4
            self._clear_reaction_context()
            return

        player = self.game.players[self.agent_index]
        tile = self._reaction_tile

        if action[0] == "ron":
            player.hand.append(tile)
            player.hand.sort()
            self.game.resolve_win(self.agent_index, "ron", tile, discarder=self._reaction_discarder)
            self._clear_reaction_context()
            return
        supplement: Optional[int] = None
        if action[0] == "kong":
            supplement = self.game.execute_open_kong(
                self.agent_index, tile, self._reaction_discarder
            )
        elif action[0] == "pon":
            if not self.game.execute_pon(
                self.agent_index, tile, self._reaction_discarder
            ):
                # Treat the failed claim as a skip and pass priority onward.
                self.game.current_player = (self._reaction_discarder + 1) % 4
                self._clear_reaction_context()
                return
        elif action[0] == "chi":
            base_tile = action[1] if action[1] is not None else tile
            self.game.execute_chi(
                self.agent_index,
                base_tile,
                self._reaction_discarder,
                self._reaction_tile,
            )
        else:
            raise ValueError(f"Unsupported reaction action {action[0]}")

        if not self.game.round_over:
            if action[0] == "kong":
                player.must_discard = False
                self._pending_draw_override = supplement
            elif action[0] == "pon":
                player.must_discard = True
            else:
                player.must_discard = True
            self.game.current_player = self.agent_index
        self._clear_reaction_context()

    def _execute_reaction(
        self,
        actor_idx: int,
        action: Tuple[str, Optional[int]],
        discarder: int,
        tile: int,
    ) -> None:
        actor = self.game.players[actor_idx]
        if action[0] == "ron":
            actor.hand.append(tile)
            actor.hand.sort()
            self.game.resolve_win(actor_idx, "ron", tile, discarder=discarder)
            return
        if action[0] == "kong":
            self.game.execute_open_kong(actor_idx, tile, discarder)
            actor.must_discard = True
            self.game.current_player = actor_idx
        elif action[0] == "pon":
            if self.game.execute_pon(actor_idx, tile, discarder):
                actor.must_discard = True
                self.game.current_player = actor_idx
            else:
                # Treat as a skipped claim and advance to the next player.
                self.game.current_player = (discarder + 1) % 4
        elif action[0] == "chi":
            base_tile = action[1] if action[1] is not None else tile
            self.game.execute_chi(actor_idx, base_tile, discarder, tile)
            actor.must_discard = True
            self.game.current_player = actor_idx

    # ----- Rendering -------------------------------------------------------

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            raise ValueError("Only human render mode is supported")
        self.game.show_game_state()

    def close(self) -> None:  # pragma: no cover - compatibility stub
        """Nothing to clean up, kept for gym compatibility."""
        return


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed()
    game = MahjongGame()
    game.play_round()


if __name__ == "__main__":
    main()

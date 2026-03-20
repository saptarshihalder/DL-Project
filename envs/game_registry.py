"""
BASIL Game Environment Registry

Provides a unified interface across TicTacToe and Connect4 so the
world model, data collector, and planning module can work with any
game without game-specific logic.

Each game environment exposes:
    - reset() -> state (H, W, 3)
    - step(action) -> (state, reward, done, info)
    - get_valid_actions() -> list[int]
    - get_state() -> np array (H, W, 3)
    - rows, cols attributes

The registry adds:
    - max_actions: maximum action space size across all games
    - action_dim: used for action embedding in Mamba world model
    - pad_state(): unify board sizes to (PAD_H, PAD_W, 3)
"""

import numpy as np
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour

# Padded board dimensions (must fit all games)
PAD_H = 7
PAD_W = 7

# Action spaces:  TTT = 9 (3x3 cells),  C4 = 7 (columns)
# Use max across all registered games for unified embedding
MAX_ACTIONS = 9  # max(9, 7) — TTT has the largest action index


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GAME_REGISTRY = {
    "tictactoe": {
        "class": TicTacToe,
        "rows": 3,
        "cols": 3,
        "num_actions": 9,      # 3x3 cell indices
        "win_length": 3,
        "description": "3x3 Tic-Tac-Toe, ~10^3 states",
    },
    "connect4": {
        "class": ConnectFour,
        "rows": 6,
        "cols": 7,
        "num_actions": 7,      # 7 column drops
        "win_length": 4,
        "description": "7x6 Connect Four, ~4.5x10^12 states",
    },
}


def make_env(game_name: str):
    """Create a game environment by name.

    Args:
        game_name: one of 'tictactoe', 'connect4'

    Returns:
        env instance with standard interface
    """
    if game_name not in GAME_REGISTRY:
        raise ValueError(
            f"Unknown game '{game_name}'. Available: {list(GAME_REGISTRY.keys())}"
        )
    return GAME_REGISTRY[game_name]["class"]()


def get_num_actions(game_name: str) -> int:
    """Return the action space size for a specific game."""
    return GAME_REGISTRY[game_name]["num_actions"]


def pad_state(state: np.ndarray, pad_h: int = PAD_H, pad_w: int = PAD_W) -> np.ndarray:
    """Pad a board state (H, W, 3) to (pad_h, pad_w, 3) with zeros."""
    h, w, c = state.shape
    if h > pad_h or w > pad_w:
        raise ValueError(f"State ({h}x{w}) exceeds pad ({pad_h}x{pad_w})")
    padded = np.zeros((pad_h, pad_w, c), dtype=np.float32)
    padded[:h, :w, :] = state
    return padded


def list_games():
    """Print all registered games."""
    for name, info in GAME_REGISTRY.items():
        print(f"  {name:12s}  {info['rows']}x{info['cols']}  "
              f"actions={info['num_actions']}  {info['description']}")

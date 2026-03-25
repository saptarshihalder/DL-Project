import os
import sys
import json
import random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W


def evaluate_agent(agent_fn, env_class, num_games=1000, device="cpu"):
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0
    fast_moves = 0
    slow_moves = 0
    game_lengths = []

    for _ in range(num_games):
        env = env_class()
        state = env.reset()
        done = False
        moves_this_game = 0

        while not done:
            if env.current_player == 1:
                valid = env.get_valid_actions()
                result = agent_fn(state, valid, device)
                if isinstance(result, tuple):
                    action, used_planning = result
                    if used_planning:
                        slow_moves += 1
                    else:
                        fast_moves += 1
                else:
                    action = result
                    fast_moves += 1
            else:
                valid = env.get_valid_actions()
                action = random.choice(valid)

            state, reward, done, info = env.step(action)
            moves_this_game += 1
            total_moves += 1

        if info["winner"] == 1:
            wins += 1
        elif info["winner"] == 2:
            losses += 1
        else:
            draws += 1
        game_lengths.append(moves_this_game)

    total = wins + losses + draws
    results = {
        "num_games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "total_moves": total_moves,
        "fast_moves": fast_moves,
        "slow_moves": slow_moves,
        "fast_pct": fast_moves / total_moves if total_moves > 0 else 0,
        "slow_pct": slow_moves / total_moves if total_moves > 0 else 0,
        "avg_game_length": np.mean(game_lengths),
    }
    return results


def evaluate_random_vs_random(env_class, num_games=1000):
    def random_agent(state, valid, device):
        return random.choice(valid)
    return evaluate_agent(random_agent, env_class, num_games)


def print_results(results, label="Agent"):
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Games:      {results['num_games']}")
    print(f"  Wins:       {results['wins']}  ({results['win_rate']:.1%})")
    print(f"  Losses:     {results['losses']}  ({results['loss_rate']:.1%})")
    print(f"  Draws:      {results['draws']}  ({results['draw_rate']:.1%})")
    print(f"  Avg length: {results['avg_game_length']:.1f} moves")
    if results['slow_moves'] > 0:
        print(f"  Fast path:  {results['fast_moves']}  ({results['fast_pct']:.1%})")
        print(f"  Slow path:  {results['slow_moves']}  ({results['slow_pct']:.1%})")
    print(f"{'=' * 50}")


def save_results(results, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {path}")


def load_results(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Random vs Random baselines:\n")

    for name, env_class in [("TicTacToe", TicTacToe), ("Connect4", ConnectFour)]:
        results = evaluate_random_vs_random(env_class, num_games=1000)
        print_results(results, f"Random vs Random — {name}")
        save_results(results, f"experiments/random_vs_random_{name.lower()}.json")

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoder_decoder import Encoder
from models.mamba_world_model import MambaWorldModel
from envs.tictactoe import TicTacToe
from envs.connect4 import ConnectFour
from envs.game_registry import pad_state, PAD_H, PAD_W, MAX_ACTIONS

LATENT_DIM = 64
ACTION_DIM = 16
HIDDEN_DIM = 128
STATE_DIM = 16
NUM_BLOCKS = 2
NUM_TRAJECTORIES = 100
HORIZON = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    encoder = Encoder(in_channels=3, latent_dim=LATENT_DIM,
                      pad_h=PAD_H, pad_w=PAD_W).to(DEVICE)
    enc_path = "checkpoints/autoencoder.pt"
    if os.path.exists(enc_path):
        ckpt = torch.load(enc_path, map_location=DEVICE, weights_only=True)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"Loaded encoder from {enc_path}")
    else:
        print(f"WARNING: No encoder checkpoint at {enc_path}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    wm = MambaWorldModel(
        latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
        max_actions=MAX_ACTIONS, hidden_dim=HIDDEN_DIM,
        state_dim=STATE_DIM, num_blocks=NUM_BLOCKS,
    ).to(DEVICE)
    wm_path = "checkpoints/world_model.pt"
    if os.path.exists(wm_path):
        ckpt = torch.load(wm_path, map_location=DEVICE, weights_only=True)
        wm.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded world model from {wm_path}")
    else:
        print(f"WARNING: No world model checkpoint at {wm_path}")
    wm.eval()

    return encoder, wm


def encode_state(encoder, state):
    padded = pad_state(state)
    x = torch.tensor(padded.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return encoder(x)


def collect_trajectory(env_class, min_length=6):
    while True:
        env = env_class()
        state = env.reset()
        states = [state.copy()]
        actions = []
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = random.choice(valid)
            actions.append(action)
            state, reward, done, info = env.step(action)
            states.append(state.copy())
        if len(actions) >= min_length:
            return states, actions


def test_multistep(encoder, wm, env_class, game_name, num_traj=NUM_TRAJECTORIES):
    mse_1step = []
    mse_5step = []

    for traj_idx in range(num_traj):
        states, actions = collect_trajectory(env_class, min_length=HORIZON + 1)

        z_real = []
        for s in states:
            z_real.append(encode_state(encoder, s))
        z_real = torch.cat(z_real, dim=0)

        max_start = len(actions) - 1
        for t in range(min(max_start, len(actions) - HORIZON)):
            wm.reset_hidden(1, DEVICE)
            a_tensor = torch.tensor([actions[t]], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                z_pred_1, r_pred, d_pred = wm.predict(z_real[t:t+1], a_tensor)
            mse_1 = nn.functional.mse_loss(z_pred_1, z_real[t+1:t+2]).item()
            mse_1step.append(mse_1)

            if t + HORIZON < len(actions):
                wm.reset_hidden(1, DEVICE)
                z_curr = z_real[t:t+1]
                for step in range(HORIZON):
                    a_t = torch.tensor([actions[t + step]], dtype=torch.long).to(DEVICE)
                    with torch.no_grad():
                        z_curr, _, _ = wm.predict(z_curr, a_t)
                mse_5 = nn.functional.mse_loss(z_curr, z_real[t+HORIZON:t+HORIZON+1]).item()
                mse_5step.append(mse_5)

    return {
        "game": game_name,
        "1step_mse_mean": np.mean(mse_1step),
        "1step_mse_std": np.std(mse_1step),
        "5step_mse_mean": np.mean(mse_5step) if mse_5step else float("nan"),
        "5step_mse_std": np.std(mse_5step) if mse_5step else float("nan"),
        "1step_samples": len(mse_1step),
        "5step_samples": len(mse_5step),
    }


def main():
    print("=" * 64)
    print("  BASIL — Multi-Step Prediction Accuracy Test")
    print("=" * 64)
    print(f"  Device: {DEVICE}")
    print(f"  Horizon: {HORIZON} steps")
    print(f"  Trajectories per game: {NUM_TRAJECTORIES}")
    print()

    encoder, wm = load_models()
    print()

    results = []
    for name, env_class in [("TicTacToe", TicTacToe), ("Connect4", ConnectFour)]:
        print(f"Testing {name}...")
        r = test_multistep(encoder, wm, env_class, name)
        results.append(r)

        print(f"  1-step MSE: {r['1step_mse_mean']:.4f} +/- {r['1step_mse_std']:.4f}  "
              f"({r['1step_samples']} samples)")
        print(f"  5-step MSE: {r['5step_mse_mean']:.4f} +/- {r['5step_mse_std']:.4f}  "
              f"({r['5step_samples']} samples)")

        if r['5step_mse_mean'] > 0 and r['1step_mse_mean'] > 0:
            ratio = r['5step_mse_mean'] / r['1step_mse_mean']
            print(f"  5-step / 1-step ratio: {ratio:.2f}x")
            if ratio < 10:
                print(f"  ACCEPTABLE — 5-step error within 10x of 1-step")
            else:
                print(f"  WARNING — 5-step error {ratio:.0f}x worse, consider more training data")
        print()

    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  {'Game':<12} {'1-step MSE':>12} {'5-step MSE':>12} {'Ratio':>8}")
    print("  " + "-" * 46)
    for r in results:
        ratio = r['5step_mse_mean'] / r['1step_mse_mean'] if r['1step_mse_mean'] > 0 else 0
        print(f"  {r['game']:<12} {r['1step_mse_mean']:>12.4f} {r['5step_mse_mean']:>12.4f} {ratio:>7.2f}x")
    print("=" * 64)


if __name__ == "__main__":
    main()

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mamba_world_model import MambaWorldModel


class WorldModelInterface:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def reset(self, batch_size=1):
        self.model.reset_hidden(batch_size, self.device)

    def predict(self, z_t, action):
        with torch.no_grad():
            z_next, reward, done = self.model.predict(
                z_t.to(self.device), action.to(self.device)
            )
        return z_next, reward, done

    def rollout(self, z_start, actions):
        with torch.no_grad():
            z_states, rewards, dones = self.model.multi_step_rollout(
                z_start.to(self.device), actions.to(self.device)
            )
        return z_states, rewards, dones


def load_world_model(checkpoint_path="checkpoints/world_model.pt", device="cuda"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Run train_world_model.py first.")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = MambaWorldModel(
        latent_dim=ckpt.get("latent_dim", 64),
        action_dim=ckpt.get("action_dim", 16),
        max_actions=ckpt.get("max_actions", 9),
        hidden_dim=ckpt.get("hidden_dim", 128),
        state_dim=ckpt.get("state_dim", 16),
        num_blocks=ckpt.get("num_blocks", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    return WorldModelInterface(model, device)


if __name__ == "__main__":
    print("Testing WorldModelInterface...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = load_world_model("checkpoints/world_model.pt", device)

    B = 1
    z = torch.randn(B, 64)
    a = torch.randint(0, 9, (B,))

    wm.reset(B)
    z_next, reward, done = wm.predict(z, a)
    print(f"  predict() -> z_next {tuple(z_next.shape)}, reward {tuple(reward.shape)}, done {tuple(done.shape)}")
    assert z_next.shape == (B, 64)
    assert reward.shape == (B,)
    assert done.shape == (B,)
    assert 0 <= done.item() <= 1

    H = 5
    actions_seq = torch.randint(0, 9, (B, H))
    wm.reset(B)
    z_states, rewards, dones = wm.rollout(z, actions_seq)
    print(f"  rollout()  -> z_states {tuple(z_states.shape)}, rewards {tuple(rewards.shape)}, dones {tuple(dones.shape)}")
    assert z_states.shape == (B, H, 64)
    assert rewards.shape == (B, H)
    assert dones.shape == (B, H)

    print("\n  ALL INTERFACE CHECKS PASSED")

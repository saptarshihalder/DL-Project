import torch
import torch.nn as nn


class GRUWorldModel(nn.Module):
    def __init__(self, latent_dim=64, action_dim=16, max_actions=9, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_embed = nn.Embedding(max_actions, action_dim)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._hidden = None

    def reset_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        self._hidden = torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(self, z_t, action, hidden=None):
        a_emb = self.action_embed(action)
        fused = torch.cat([z_t, a_emb], dim=-1)
        h = self.fusion(fused).unsqueeze(1)
        h_state = hidden if hidden is not None else self._hidden
        out, h_new = self.gru(h, h_state)
        self._hidden = h_new
        out = out.squeeze(1)
        z_next = self.next_state_head(out)
        reward = self.reward_head(out)
        done_logit = self.done_head(out)
        return z_next, reward, done_logit, h_new

    def predict(self, z_t, action):
        z_next, reward, done_logit, _ = self.forward(z_t, action)
        return z_next, reward.squeeze(-1), torch.sigmoid(done_logit).squeeze(-1)

    def multi_step_rollout(self, z_start, actions):
        B, H = actions.shape
        self.reset_hidden(B, z_start.device)
        z_states, rewards, dones = [], [], []
        z_t = z_start
        for t in range(H):
            z_t, r, d = self.predict(z_t, actions[:, t])
            z_states.append(z_t)
            rewards.append(r)
            dones.append(d)
        return torch.stack(z_states, 1), torch.stack(rewards, 1), torch.stack(dones, 1)


class LSTMWorldModel(nn.Module):
    def __init__(self, latent_dim=64, action_dim=16, max_actions=9, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_embed = nn.Embedding(max_actions, action_dim)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self._h = None
        self._c = None

    def reset_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        self._h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        self._c = torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(self, z_t, action, hidden=None):
        a_emb = self.action_embed(action)
        fused = torch.cat([z_t, a_emb], dim=-1)
        h = self.fusion(fused).unsqueeze(1)
        if hidden is not None:
            h_state = hidden
        else:
            h_state = (self._h, self._c)
        out, (h_new, c_new) = self.lstm(h, h_state)
        self._h = h_new
        self._c = c_new
        out = out.squeeze(1)
        z_next = self.next_state_head(out)
        reward = self.reward_head(out)
        done_logit = self.done_head(out)
        return z_next, reward, done_logit, (h_new, c_new)

    def predict(self, z_t, action):
        z_next, reward, done_logit, _ = self.forward(z_t, action)
        return z_next, reward.squeeze(-1), torch.sigmoid(done_logit).squeeze(-1)

    def multi_step_rollout(self, z_start, actions):
        B, H = actions.shape
        self.reset_hidden(B, z_start.device)
        z_states, rewards, dones = [], [], []
        z_t = z_start
        for t in range(H):
            z_t, r, d = self.predict(z_t, actions[:, t])
            z_states.append(z_t)
            rewards.append(r)
            dones.append(d)
        return torch.stack(z_states, 1), torch.stack(rewards, 1), torch.stack(dones, 1)


class MLPWorldModel(nn.Module):
    def __init__(self, latent_dim=64, action_dim=16, max_actions=9, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_embed = nn.Embedding(max_actions, action_dim)
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def reset_hidden(self, batch_size, device=None):
        pass

    def forward(self, z_t, action, hidden=None):
        a_emb = self.action_embed(action)
        fused = torch.cat([z_t, a_emb], dim=-1)
        h = self.fusion(fused)
        h = self.mlp(h)
        z_next = self.next_state_head(h)
        reward = self.reward_head(h)
        done_logit = self.done_head(h)
        return z_next, reward, done_logit, None

    def predict(self, z_t, action):
        z_next, reward, done_logit, _ = self.forward(z_t, action)
        return z_next, reward.squeeze(-1), torch.sigmoid(done_logit).squeeze(-1)

    def multi_step_rollout(self, z_start, actions):
        B, H = actions.shape
        z_states, rewards, dones = [], [], []
        z_t = z_start
        for t in range(H):
            z_t, r, d = self.predict(z_t, actions[:, t])
            z_states.append(z_t)
            rewards.append(r)
            dones.append(d)
        return torch.stack(z_states, 1), torch.stack(rewards, 1), torch.stack(dones, 1)


MODEL_REGISTRY = {
    "gru": GRUWorldModel,
    "lstm": LSTMWorldModel,
    "mlp": MLPWorldModel,
}


def make_baseline(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)

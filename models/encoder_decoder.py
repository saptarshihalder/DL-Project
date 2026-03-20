import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64,
                 pad_h: int = 7, pad_w: int = 7):
        super().__init__()
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.latent_dim = latent_dim

        # Three Conv2D layers: C -> 32 -> 64 -> 64 (paper Section 3)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Flatten and project to latent_dim
        self.flat_dim = 64 * pad_h * pad_w  # 64 * 7 * 7 = 3136
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_layers(x)           # (B, 64, 7, 7)
        h = h.reshape(h.size(0), -1)      # (B, 3136)
        z = self.fc(h)                     # (B, 64)
        return z


class Decoder(nn.Module):

    def __init__(self, out_channels: int = 3, latent_dim: int = 64,
                 pad_h: int = 7, pad_w: int = 7):
        super().__init__()
        self.pad_h = pad_h
        self.pad_w = pad_w

        self.flat_dim = 64 * pad_h * pad_w  # 3136
        self.fc = nn.Linear(latent_dim, self.flat_dim)

        # Mirror of encoder conv layers (channels reversed)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, 64) latent vector
        Returns:
            x_hat: (batch, 3, 7, 7) reconstructed board
        """
        h = self.fc(z)                                        # (B, 3136)
        h = h.reshape(h.size(0), 64, self.pad_h, self.pad_w)  # (B, 64, 7, 7)
        x_hat = self.deconv_layers(h)                          # (B, 3, 7, 7)
        return x_hat


class BoardAutoEncoder(nn.Module):

    def __init__(self, in_channels: int = 3, latent_dim: int = 64,
                 pad_h: int = 7, pad_w: int = 7):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim, pad_h, pad_w)
        self.decoder = Decoder(in_channels, latent_dim, pad_h, pad_w)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, 3, 7, 7) padded board
        Returns:
            x_hat: (batch, 3, 7, 7) reconstruction
            z:     (batch, 64) latent representation
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Board -> latent. Used at inference by the rest of BASIL."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent -> board reconstruction."""
        return self.decoder(z)

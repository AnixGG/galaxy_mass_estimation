"""FNO-модели регрессии массы по трёхканальной карте и красному смещению."""

from .spectral import FNO2d
import torch
import torch.nn as nn


class ModelFNO1(nn.Module):
    """FNO с полносвязной головой; основа экспериментов base, wide и large."""

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        modes1=16,
        modes2=16,
        width=64,
        depth=4,
        scalar_dim=1,
        hidden_dim=128,
    ):
        super().__init__()

        self.fno = FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            depth=depth,
        )
        fno_out_dim = 21904

        self.regressor = nn.Sequential(
            nn.Linear(fno_out_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, img, scalars):
        x_img = self.fno(img)
        x_img = x_img.view(x_img.size(0), -1)
        x = torch.cat([x_img, scalars], dim=1)
        out = self.regressor(x)
        return out.squeeze(1)


class ModelFNO2(nn.Module):
    """FNO с LayerNorm и более глубокой головой; эксперименты less."""

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        modes1=12,
        modes2=12,
        width=48,
        depth=3,
        scalar_dim=1,
        hidden_dim=256,
    ):
        super().__init__()

        self.fno = FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            depth=depth,
        )

        fno_out_dim = 21904

        self.regressor = nn.Sequential(
            nn.Linear(fno_out_dim + scalar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, img, scalars):
        x_img = self.fno(img)
        x_img = x_img.view(x_img.size(0), -1)
        x = torch.cat([x_img, scalars], dim=1)
        out = self.regressor(x)
        return out.squeeze(1)

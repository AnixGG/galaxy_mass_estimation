"""
Fourier Neural Operator (FNO) implementation in PyTorch.
Includes:
- SpectralConv2d: Fourier layer that transforms to frequency domain, applies learnable complex weights on low-frequency modes and returns to spatial domain.
- FNO2d: A small FNO model built from SpectralConv2d layers and local convolutional blocks.

A minimal forward-pass example is provided at the end of the module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D Fourier layer that performs multiplication in frequency domain on low-frequency modes.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        modes1 (int): number of Fourier modes to keep along height
        modes2 (int): number of Fourier modes to keep along width
    """

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # We store complex weights as real tensors with last dimension 2: (real, imag)
        self.scale = 1 / (in_channels * out_channels)
        weight_shape = (in_channels, out_channels, modes1, modes2, 2)
        self.weights = nn.Parameter(self.scale * torch.randn(*weight_shape))

    def compl_mul2d(self, input_ft, weights):
        """Complex multiplication: (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
        input_ft: complex tensor represented as real/imag pair (batch, in_ch, h, w, 2)
        weights: (in_ch, out_ch, modes1, modes2, 2)
        returns: (batch, out_ch, h, w, 2) with applied low modes (others zero)
        """
        # input_ft: (..., 2), weights: (..., 2)
        a = input_ft[..., 0]
        b = input_ft[..., 1]
        c = weights[..., 0]
        d = weights[..., 1]
        # broadcast multiply: sum over in_channels
        # result real = a*c - b*d, imag = a*d + b*c
        # input_ft shape: (batch, in_ch, h, w, 2) but we'll slice modes region
        real = torch.einsum("bixy,ioxy->boxy", a, c) - torch.einsum(
            "bixy,ioxy->boxy", b, d
        )
        imag = torch.einsum("bixy,ioxy->boxy", a, d) + torch.einsum(
            "bixy,ioxy->boxy", b, c
        )
        return torch.stack([real, imag], dim=-1)

    def forward(self, x):
        """x: (batch, in_channels, height, width)"""
        batchsize, channels, height, width = x.shape
        # Compute Fourier coeffs
        x_ft = torch.fft.rfft2(
            x, norm="ortho"
        )  # shape: (batch, in_ch, h, w//2+1), complex

        # Convert to real/imag stacked for easier parameter handling
        x_ft = torch.view_as_real(x_ft)  # (batch, in_ch, h, w2, 2)

        # allocate output in frequency domain (real/imag)
        out_ft = torch.zeros(
            (batchsize, self.out_channels, height, x_ft.shape[3], 2),
            device=x.device,
            dtype=x_ft.dtype,
        )

        # low frequency block: top-left corner
        m1 = min(self.modes1, height)
        m2 = min(self.modes2, x_ft.shape[3])

        # slice the low modes and multiply
        input_slice = x_ft[:, :, :m1, :m2, :]  # (batch, in_ch, m1, m2, 2)
        weights = self.weights[:, :, :m1, :m2, :]  # (in_ch, out_ch, m1, m2, 2)

        # move in_ch to axis 1 for einsum compat if needed; our compl_mul expects in_ch at dim=1
        # currently shapes match for einsum formulation in compl_mul2d
        out_low = self.compl_mul2d(input_slice, weights)
        # out_low: (batch, out_ch, m1, m2, 2)

        out_ft[:, :, :m1, :m2, :] = out_low

        # Back to complex tensor and inverse FFT
        out_ft_complex = torch.view_as_complex(out_ft)  # (batch, out_ch, h, w2)
        x = torch.fft.irfft2(out_ft_complex, s=(height, width), norm="ortho")
        # result: (batch, out_ch, height, width) real
        return x


class FNO2d(nn.Module):
    """A simple Fourier Neural Operator for 2D inputs.

    Architecture outline:
        - input lifting (1x1 conv) -> width
        - several layers: SpectralConv2d + pointwise conv + activation
        - projection to output via 1x1 convs

    Args:
        modes1, modes2: number of Fourier modes to keep
        width: channel width inside the FNO
        depth: number of Fourier layers
    """

    def __init__(self, in_channels, out_channels, modes1, modes2, width=64, depth=4):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth

        # input lifting
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)

        # spectral layers
        self.spectral_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)]
        )

        # 1x1 convolutions (local part) to complement spectral conv
        self.w_convs = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)]
        )

        # activation
        self.activation = F.gelu

        # projection to output
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width // 2, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # x: (batch, in_channels, height, width)
        x = self.input_proj(x)

        for spec, w in zip(self.spectral_layers, self.w_convs):
            x1 = spec(x)
            x2 = w(x)
            x = self.activation(x1 + x2)

        x = self.output_proj(x)
        return x


if __name__ == "__main__":
    batch = 2
    in_ch = 3
    out_ch = 1
    H = 64
    W = 64

    model = FNO2d(
        in_channels=in_ch, out_channels=out_ch, modes1=16, modes2=16, width=64, depth=4
    )
    x = torch.randn(batch, in_ch, H, W)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

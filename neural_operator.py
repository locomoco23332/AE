import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    """1D Fourier spectral convolution layer used in FNO."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication."""
        # input: (batch, in_channels, modes)
        # weights: (in_channels, out_channels, modes)
        return torch.einsum("bim, iom -> bom", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # x: (batch, in_channels, n)
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.size(-1),
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes] = self.compl_mul1d(
            x_ft[:, :, : self.modes], self.weights
        )
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    """Simple Fourier Neural Operator for 1D inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int,
        modes: int,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.fc0 = nn.Linear(in_channels, width)

        self.convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(depth)]
        )
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(depth)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n, in_channels)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # (batch, width, n)
        for conv, w, bn in zip(self.convs, self.ws, self.norms):
            x = bn(F.gelu(conv(x) + w(x)))
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbeddings(nn.Module):
    def __init__(self, d_model: int, patch_size: int):
        super().__init__()

        self.conv = nn.Conv3d(in_channels=1, out_channels=d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv(x)

        batch_size, c, d, h, w = x.shape

        x = x.permute(0, 2, 3, 4, 1)
        x = x.view(batch_size, d * h * w, c)

        return x


class UnpatchEmbeddings(nn.Module):
    def __init__(self, d_model: int, patch_size: int, shape: Tuple[int, int, int]):
        super().__init__()

        self.conv = nn.ConvTranspose3d(in_channels=d_model, out_channels=1, kernel_size=patch_size, stride=patch_size)
        self.depth, self.height, self.width = shape
        self.depth //= patch_size
        self.height //= patch_size
        self.width //= patch_size

    def forward(self, x: torch.Tensor):
        batch_size, dhw, c = x.shape
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, c, self.depth, self.height, self.width)
        x = self.conv(x)

        return x


class LearnedPositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5_000):
        super().__init__()

        self.position_encodings = nn.Parameter(torch.zeros(max_len, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        batch_size, length, d_model = x.shape
        pe = self.position_encodings.repeat(batch_size, 1, 1)

        return x + pe


class CentralCutout(nn.Module):
    def __init__(self, new_shape: Tuple[int, int, int]):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x: torch.Tensor):
        """
        Extract a centered subvolume from a 5D tensor.
        Args:
        - tensor (torch.Tensor): Input tensor of shape (batch_size, 1, depth, height, width).

        Returns:
        - torch.Tensor: Cutout tensor of shape (batch_size, new_depth, new_height, new_width).
        """
        batch_size, channels, depth, height, width = x.shape

        new_depth, new_height, new_width = self.new_shape

        # Calculate the start and end indices
        start_depth = (depth - new_depth) // 2
        end_depth = start_depth + new_depth
        start_height = (height - new_height) // 2
        end_height = start_height + new_height
        start_width = (width - new_width) // 2
        end_width = start_width + new_width

        # Extract the centered subvolume
        return x[:, 0, start_depth:end_depth, start_height:end_height, start_width:end_width]


class VisionTransformer(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], output_shape: Tuple[int, int, int], d_model: int, n_heads=8,
                 n_layers=12, dim_feedforward=2048, patch_size=4, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.depth, self.height, self.width = shape
        self.patch_size = patch_size
        self.output_shape = output_shape

        self.n_patches = self.width // patch_size

        assert self.depth % patch_size == 0, "voxel depth not divisible by patch size"
        assert self.height % patch_size == 0, "voxel height not divisible by patch size"
        assert self.width % patch_size == 0, "voxel width not divisible by patch size"

        self.patch_embed = PatchEmbeddings(d_model, patch_size)

        self.position_embed = LearnedPositionalEmbeddings(d_model, self.n_patches ** 3)

        self.encode = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="relu"
            )
            for _ in range(n_layers)
        ])

        self.unpatch_embed = UnpatchEmbeddings(d_model, patch_size, shape)

        self.cutout = CentralCutout(output_shape)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, depth, height, width = x.shape

        assert height % self.patch_size == 0, "voxel height not divisible by patch size"
        assert width % self.patch_size == 0, "voxel width not divisible by patch size"
        assert depth % self.patch_size == 0, "voxel depth not divisible by patch size"

        x = self.patch_embed(x)

        x = self.position_embed(x)

        x = self.encode(x)

        x = self.unpatch_embed(x)

        x = self.cutout(x)

        x = self.sigmoid(x)

        return x

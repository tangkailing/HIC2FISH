

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import MATRIX_SIZE


class DoubleConv(nn.Module):
    """Two 3 x 3 convolutions, each followed by GroupNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class EnhancedUNet(nn.Module):
    """Hi-C-conditioned U-Net that predicts diffusion noise."""

    def __init__(self) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        self.down1 = DoubleConv(2, 64)
        self.down2_pool = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.down3_pool = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)

        self.t_proj = nn.Conv2d(64, 64, kernel_size=1)

        self.up1_deconv = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.up1 = DoubleConv(256, 128)

        self.attn_proj = nn.Linear(128, 128)
        self.attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            batch_first=True,
        )

        self.up2_deconv = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.up2 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    @staticmethod
    def center_crop_to_match(
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        _, _, source_height, source_width = source.shape
        _, _, target_height, target_width = target.shape
        height_difference = source_height - target_height
        width_difference = source_width - target_width
        if height_difference < 0 or width_difference < 0:
            raise ValueError(
                f"Cannot crop source {tuple(source.shape)} "
                f"to target {tuple(target.shape)}."
            )
        return source[
            :,
            :,
            height_difference
            // 2 : source_height
            - (height_difference - height_difference // 2),
            width_difference
            // 2 : source_width
            - (width_difference - width_difference // 2),
        ]

    def forward(
        self,
        noisy_matrix: torch.Tensor,
        condition: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = self.time_mlp(diffusion_time).view(-1, 64, 1, 1)
        time_embedding = F.interpolate(
            time_embedding,
            size=noisy_matrix.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        time_embedding = self.t_proj(time_embedding)

        inputs = torch.cat([noisy_matrix, condition], dim=1)
        encoder_1 = self.down1(inputs) + time_embedding
        encoder_2 = self.down2(self.down2_pool(encoder_1))
        bottleneck = self.down3(self.down3_pool(encoder_2))

        decoded = self.up1_deconv(bottleneck)
        encoder_2 = self.center_crop_to_match(encoder_2, decoded)
        decoded = self.up1(torch.cat([decoded, encoder_2], dim=1))

        batch_size, channels, height, width = decoded.shape
        attention = decoded.view(batch_size, channels, -1).permute(0, 2, 1)
        attention = self.attn_proj(attention)
        attention, _ = self.attn(attention, attention, attention)
        decoded = attention.permute(0, 2, 1).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        decoded = self.up2_deconv(decoded)
        encoder_1 = self.center_crop_to_match(encoder_1, decoded)
        decoded = self.up2(torch.cat([decoded, encoder_1], dim=1))
        predicted_noise = self.final(decoded)
        return F.interpolate(
            predicted_noise,
            size=(MATRIX_SIZE, MATRIX_SIZE),
            mode="bilinear",
            align_corners=False,
        )

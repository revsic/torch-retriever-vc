from typing import Optional

import torch
import torch.nn as nn

from ..transformer import AddNorm, FeedForward, MultiheadAttention


class LinkAttention(nn.Module):
    """Link attention to retrieve content-specific style for reconstruction.
    """
    def __init__(self,
                 kernels: int,
                 channels: int,
                 heads: int,
                 stylelen: int,
                 blocks: int):
        """Initializer.
        Args:
            kernels: size of the convolutional kernels.
            channels: size of the input channels.
            heads: the number of the attention heads.
            stylelen: the number of the style vectors.
            blocks: the number of the attention blocks.
        """
        super().__init__()
        # depthwise convolution
        self.conv = nn.Conv1d(
            channels, channels, kernels, padding=kernels // 2, groups=channels)
        self.linkkey = nn.Parameter(nn.randn(1, stylelen, channels))
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                AddNorm(channels, MultiheadAttention(channels, heads)),
                nn.Sequential(
                    AddNorm(channels, MultiheadAttention(channels, heads)),
                    AddNorm(channels, FeedForward(channels)))])
            for _ in range(blocks)])

    def forward(self,
                contents: torch.Tensor,
                style: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """Retrieve the content-specific styles.
        Args:
            contents: [torch.float32; [B, T, C]], content vectors.
            style: [torch.float32; [B, S, C]], style vectors.
            mask: [torch.float32; [B, T]], mask vector.
        Returns:
            [torch.float32; [B, T, C]], retrieved.
        """
        # [B, T, C]
        x = self.conv(contents.transpose(1, 2)).transpose(1, 2)
        if mask is not None:
            # [B, T, 1]
            mask = mask[..., None]
            # [B, T, C]
            contents = contents * mask
            # [B, T, T]
            mask_self = mask * mask.transpose(1, 2)
        for selfattn, linkattn in self.blocks:
            # [B, T, C]
            x = selfattn(x, x, x, mask=mask_self)
            # [B, T, C]
            x = linkattn(x, self.linkkey, style, mask=mask)
            if mask is not None:
                # [B, T, C], masking for FFN.
                x = x * mask
        # [B, T, C]
        return x

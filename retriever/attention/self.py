from typing import Optional

import torch
import torch.nn as nn

from ..transformer import AddNorm, FeedForward, MultiheadAttention


class SelfAttention(nn.Module):
    """Self-attention.
    """
    def __init__(self, channels: int, heads: int, blocks: int):
        """Initializer.
            channels: size of the input channels.
            heads: the number of the attention heads.
            blocks: the number of the attention blocks.
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                AddNorm(channels, MultiheadAttention(channels, heads)),
                AddNorm(channels, FeedForward(channels)))
            for _ in range(blocks)])

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.tensor] = None) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, T, C]], input tensor.
            mask: [torch.float32; [B, T]], sequence mask.
        Returns:
            [torch.float32; [B, T, C]], transformed.
        """
        # [B, T, C]
        x = inputs
        if mask is not None:
            # [B, T, T]
            mask = mask[..., None] * mask[:, None]
        for block in self.blocks:
            # [B, T, C]
            x = block(x, x, x, mask=mask)
            if mask is not None:
                # [B, T, C]
                x = x * mask[..., 0:1]
        # [B, T, C]
        return x

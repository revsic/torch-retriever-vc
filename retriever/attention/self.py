from typing import Optional

import torch
import torch.nn as nn

from ..transformer import AddNorm, FeedForward, MultiheadAttention, SequentialWrapper


class SelfAttention(nn.Module):
    """Self-attention.
    """
    def __init__(self,
                 channels: int,
                 heads: int,
                 ffn: int,
                 blocks: int,
                 dropout: float = 0.):
        """Initializer.
            channels: size of the input channels.
            heads: the number of the attention heads.
            ffn: size of the FFN hidden channels.
            blocks: the number of the attention blocks.
            dropout: dropout rates for FFN.
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            SequentialWrapper(
                AddNorm(channels, MultiheadAttention(channels, channels, channels, heads)),
                AddNorm(channels, FeedForward(channels, ffn, dropout)))
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

from typing import Optional

import torch
import torch.nn as nn

from ..transformer import AddNorm, FeedForward, MultiheadAttention, SequentialWrapper


class CrossAttention(nn.Module):
    """Cross-attention to retrieve style from input data.
    """
    def __init__(self, channels: int, heads: int, stylelen: int, blocks: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            heads: the number of the attention heads.
            stylelen: the number of the style vectors.
            blocks: the number of the attention blocks.
        """
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(1, stylelen, channels))
        self.blocks = nn.ModuleList([
            SequentialWrapper(
                AddNorm(channels, MultiheadAttention(channels, heads)),
                AddNorm(
                    channels,
                    nn.Sequential(
                        nn.Conv1d(stylelen, stylelen, 1),
                        nn.ReLU(),
                        nn.Conv1d(stylelen, stylelen, 1))),
                AddNorm(channels, FeedForward(channels)))
            for _ in range(blocks)])

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate the style tokens.
        Args:
            inputs: [torch.float32; [B, T, C]], input tensor.
            prototype: [torch.float32; [B, S, C]], prototype style tokens.
            mask: [torch.float32; [B, T]], input mask, if provided.
        Returns:
            [torch.float32; [B, S, C]], retrieved style tokens.
        """
        if mask is not None:
            # [B, 1, T]
            mask = mask[:, None]
        # [B, S, C]
        style = self.prototypes.repeat(inputs.shape[0], 1, 1)
        for block in self.blocks:
            # [B, S, C]
            style = block(style, inputs, inputs, mask=mask)
        return style

from typing import Optional

import torch
import torch.nn as nn

from .transformer import AddNorm, AuxSequential, FeedForward, MultiheadAttention


class CrossAttention(nn.Module):
    """Cross-attention to retrieve style from input data.
    """
    def __init__(self,
                 contexts: int,
                 styles: int,
                 heads: int,
                 ffn: int,
                 prototypes: int,
                 blocks: int,
                 dropout: float = 0.):
        """Initializer.
        Args:
            contexts: size of the context channels.
            styles: size of the style channels.
            heads: the number of the attention heads.
            ffn: size of the FFN hidden channels.
            prototypes: the number of the style vectors.
            blocks: the number of the attention blocks.
            dropout: dropout rates for FFN.
        """
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(1, prototypes, styles))
        self.blocks = nn.ModuleList([
            AuxSequential(
                AddNorm(styles, MultiheadAttention(contexts, styles, styles, heads)),
                AddNorm(
                    styles,
                    nn.Sequential(
                        nn.Conv1d(prototypes, prototypes, 1),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(prototypes, prototypes, 1))),
                AddNorm(styles, FeedForward(styles, ffn, dropout)))
            for _ in range(blocks)])

    def forward(self,
                inputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate the style tokens.
        Args:
            inputs: [torch.float32; [B, T, contexts]], input tensor.
            mask: [torch.float32; [B, T]], input mask, if provided.
        Returns:
            [torch.float32; [B, prototypes, styles]], retrieved style tokens.
        """
        if mask is not None:
            # [B, 1, T]
            mask = mask[:, None]
        # [B, S, C]
        style = self.prototypes
        for block in self.blocks:
            # [B, S, C]
            style = block(style, inputs, inputs, mask=mask)
        return style

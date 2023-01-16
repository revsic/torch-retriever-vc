from typing import Optional

import torch
import torch.nn as nn

from .transformer import AddNorm, AuxSequential, FeedForward, MultiheadAttention


class LinkAttention(nn.Module):
    """Link attention to retrieve content-specific style for reconstruction.
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
            contexts: size of the input channels.
            styles: size of the style channels.
            heads: the number of the attention heads.
            ffn: size of the FFN hidden channels.
            prototypes: the number of the style vectors.
            blocks: the number of the attention blocks.
            dropout: dropout rates for FFN.
        """
        super().__init__()
        self.linkkey = nn.Parameter(torch.randn(1, prototypes, styles))
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                # self attention
                AddNorm(contexts, MultiheadAttention(contexts, contexts, contexts, heads)),
                # link attention
                AuxSequential(
                    AddNorm(contexts, MultiheadAttention(styles, contexts, contexts, heads)),
                    AddNorm(contexts, FeedForward(contexts, ffn, dropout)))])
            for _ in range(blocks)])

    def forward(self,
                contents: torch.Tensor,
                style: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Retrieve the content-specific styles.
        Args:
            contents: [torch.float32; [B, T, contexts]], content vectors.
            style: [torch.float32; [B, prototypes, styles]], style vectors.
            mask: [torch.float32; [B, T]], mask vector.
        Returns:
            [torch.float32; [B, T, contexts]], linked.
        """
        x = contents
        if mask is not None:
            # [B, T, T]
            mask = mask[:, None] * mask[..., None]
        # [B, S, C]
        linkkey = self.linkkey.repeat(x.shape[0], 1, 1)
        for selfattn, linkattn in self.blocks:
            # [B, T, C]
            x = selfattn(x, x, x, mask=mask)
            # [B, T, C]
            x = linkattn(x, linkkey, style, mask=mask[..., :1])
        if mask is not None:
            # [B, T, C], masking for FFN.
            x = x * mask[..., :1]
        # [B, T, C]
        return x

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AuxSequential, SelfAttention


class LinguisticEncoder(nn.Module):
    """Linguistic encoder.
    """
    def __init__(self,
                 channels: int,
                 hiddens: int,
                 heads: int,
                 ffn: int,
                 blocks: int,
                 dropout: float = 0.):
        """Initializer.
        Args:
            channels: size of the input channels.
            hiddens: size of the hidden channels.
            heads: the number of the attention heads.
            ffn: size of the FFN hidden channels.
            blocks: the number of the attention blocks.
            dropout: dropout rates for FFN.
        """
        super().__init__()
        self.proj = nn.Linear(channels, hiddens)
        self.encoder = AuxSequential(
            SelfAttention(hiddens, heads, ffn, blocks, dropout),
            nn.Linear(hiddens, hiddens))

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract the linguistic informations.
        Args:
            inputs: [torch.float32; [B, S, channels]], input tensors.
            mask: [torch.float32; [B, S]], sequential mask.
        Returns:
            [torch.float32; [B, S, hiddens]], extracted.
        """
        # [B, S, hiddens]
        x = self.proj(inputs)
        # [B, S, hiddens]
        out = F.normalize(self.encoder(x, mask=mask), dim=-1)
        if mask is not None:
            out = out * mask[..., None]
        return out

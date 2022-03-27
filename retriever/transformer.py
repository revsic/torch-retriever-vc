from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    """Multihead scaled dot-product attention.
    """
    def __init__(self, channels: int, heads: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            heads: the number of the attnetion heads.
        """
        super().__init__()
        self.channels, self.heads = channels // heads, heads
        self.proj_key = nn.Linear(channels, channels)
        self.proj_query = nn.Linear(channels, channels)
        self.proj_value = nn.Linear(channels, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self,
                key: torch.Tensor,
                query: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor):
        """Transform the inputs.
        Args:
            key: [torch.float32; [B, S, C]], key.
            query: [torch.float32; [B, T, C]], query.
            value: [torch.float32; [B, S, C]], value.
            mask: [torch.float32; [B, T, S]], attention mask.
        """
        # B, S
        bsize, keylen, _ = key.shape
        # T
        querylen = query.shape[1]
        # [B, S, H, C // H]
        key = self.proj_key(key).view(
            bsize, keylen, self.heads, self.channels)
        value = self.proj_value(value).view(
            bsize, keylen, self.heads, self.channels)
        # [B, T, H, C // H]
        query = self.proj_query(query).view(
            bsize, querylen, self.heads, self.channels)
        # [B, H, T, S]
        score = torch.matmul(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 3, 1)) * (self.channels ** -0.5)
        score.masked_fill_(~mask[:, None, 0:1].to(torch.bool), -np.inf)
        weights = torch.softmax(score, dim=-1)
        # [B, H, T, C // H]
        out = torch.matmul(weights, value.transpose(1, 2))
        # [B, T, C]
        return self.proj_out(out.transpose(1, 2).view(bsize, querylen, -1))


class FeedForward(nn.Sequential):
    """Feed-forward network.
    """
    def __init__(self, channels: int, factor: int = 4):
        """Initializer.
        Args:
            channels: size of the input channels.
            factor: upsample rates.
        """
        super().__init__(
            nn.Linear(channels, channels * factor),
            nn.ReLU(),
            nn.Linear(channels * factor, channels))


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encodings.
    """
    def __init__(self, pe: int, steps: int = 100):
        """Initializer.
        Args:
            pe: size of the positional encodings.
            steps: initial size.
        """
        super().__init__()
        self.register_buffer('buffer', SinusoidalPE.compute(steps, pe))

    def forward(self, steps: int) -> torch.Tensor:
        """Return the cached positional encodings..
        Args:
            steps: size of the pe.
        Returns:
            [torch.float32; [steps, pe]], embedding.
        """
        # S, P
        maxsteps, pe = self.buffer.shape
        if steps > maxsteps:
            self.register_buffer(
                'buffer', SinusoidalPE.compute(steps, pe, device=self.buffer.device))
        # [steps, pe]
        return self.buffer[:steps]

    @staticmethod
    def compute(steps: int, pe: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate sinusoidal embedding introduced by Vaswani et al., 2017.
        Args:
            steps: size of the pe.
            pe: size of the sinusoidal positional encodings.
        Returns:
            [torch.float32; [steps, pe]], sinusoidal positional embedding.
        """
        device = device or torch.device('cpu')
        # [S]
        pos = torch.arange(steps, device=device)
        # [C // 2]
        i = torch.arange(0, pe, 2, device=device)
        # [S, C // 2]
        context = pos[:, None] * torch.exp(-np.log(10000) * i / pe)[None]
        # [S, C]
        return torch.stack(
                [torch.sin(context), torch.cos(context)], dim=-1
            ).reshape(steps, pe)

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class AddNorm(nn.Module):
    """Add and normalization
    """
    def __init__(self, channels: int, sublayer: nn.Module):
        """Initializer.
        Args:
            channels: size of the input channels.
            sublayer: Sub-layer.
        """
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)
        self.sublayer = sublayer

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Transform, add and normalize.
        Args:
            inputs: [torch.float32; [...]], input tensor.
        Returns:
            [torch.float32; [...]], residually connected.
        """
        return self.layernorm(inputs + self.sublayer(inputs, *args, **kwargs))


class AuxSequential(nn.Module):
    """Sequential wrapper for multiple input sequential.
    """
    def __init__(self, *args):
        """Initializer.
        """
        super().__init__()
        self.fst, *rst = args
        self.rst = nn.Sequential(*rst)

    def forward(self, *args, **kwargs):
        """Multiple input wrapper.
        """
        return self.rst(self.fst(*args, **kwargs))


class MultiheadAttention(nn.Module):
    """Multihead scaled dot-product attention.
    """
    def __init__(self,
                 contexts: int,
                 queries: int,
                 channels: int,
                 heads: int):
        """Initializer.
        Args:
            contexts: size of the key, value channels.
            queries: size of the query channels.
            channels: size of the hidden channels.
            heads: the number of the attnetion heads.
        """
        super().__init__()
        self.channels, self.heads = channels // heads, heads
        self.proj_key = nn.Linear(contexts, channels)
        self.proj_value = nn.Linear(contexts, channels)
        self.proj_query = nn.Linear(queries, channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                causal: bool = False) -> torch.Tensor:
        """Transform the inputs.
        Args:
            query: [torch.float32; [B, S, queries]], query.
            key: [torch.float32; [B, T, contexts]], key.
            value: [torch.float32; [B, T, contexts]], value.
            mask: [torch.float32; [B, S, T]], attention mask, box form.
            causal: whether mask the inputs in causal form or not.
        Returns:
            [torch.float32; [B, S, C]], attended.
        """
        # S, T
        querylen, keylen = query.shape[1], key.shape[1]
        # [B, T, H, C // H]
        key = self.proj_key(key).view(
            -1, keylen, self.heads, self.channels)
        value = self.proj_value(value).view(
            -1, keylen, self.heads, self.channels)
        # [B, S, H, C // H]
        query = self.proj_query(query).view(
            -1, querylen, self.heads, self.channels)
        # [B, H, S, T]
        score = torch.matmul(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 3, 1)) * (self.channels ** -0.5)
        if mask is not None:
            # [B, _, T]
            mask_s = mask + torch.full_like(mask, -np.inf).triu(diagonal=1) \
                if causal \
                else mask[:, :1]
            # masking
            score.masked_fill_(~mask_s[:, None].to(torch.bool), -np.inf)
        weights = torch.softmax(score, dim=-1)
        # [B, H, S, C // H]
        out = torch.matmul(weights, value.transpose(1, 2))
        # B
        bsize, *_ = out.shape
        # [B, S, C]
        out = self.proj_out(out.transpose(1, 2).reshape(bsize, querylen, -1))
        if mask is not None:
            # [B, S, C]
            out = out * mask[..., :1]
        return out


class FeedForward(nn.Sequential):
    """Feed-forward network.
    """
    def __init__(self, channels: int, hiddens: int, dropout: float = 0.):
        """Initializer.
        Args:
            channels: size of the input channels.
            hiddens: size of the hidden channels.
            dropout: dropout rates.
        """
        super().__init__(
            nn.Linear(channels, hiddens),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiddens, channels))


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
        self.register_buffer(
            'buffer',
            SinusoidalPE.compute(steps, pe),
            persistent=False)

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
                'buffer',
                SinusoidalPE.compute(steps, pe, device=self.buffer.device),
                persistent=False)
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

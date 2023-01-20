from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.transformer import \
    AddNorm, AuxSequential, FeedForward, MultiheadAttention, SinusoidalPE


class MultiLink(nn.Module):
    """Attention with multiple kinds.
    """
    def __init__(self,
                 channels: int,
                 contexts: int,
                 styles: int,
                 heads: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            contexts: size of the context channels.
            styles: size of the style channels.
            heads: size of the FFN hidden channels.
        """
        super().__init__()
        self.proj = nn.Linear(channels, channels * 2)
        self.link_style = MultiheadAttention(
            styles, channels, channels, heads)
        self.link_context = MultiheadAttention(
            contexts, channels, channels, heads)

    def forward(self,
                x: torch.Tensor,
                key: torch.Tensor,
                style: torch.Tensor,
                context: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Link the information.
        Args:
            x: [torch.float32; [B, T, channels]], input query.
            key: [torch.float32; [B, prototypes, styles]], link key.
            style: [torch.float32; [B, prototypes, styles]], style vector.
            context: [torch.float32; [B, S, contexts]], context vector.
            mask: [torch.float32; [B, T, S]], attention mask.
        Returns:
            [torch.float32; [B, T, channels]], attended.
        """
        # [B, T, channels]
        x_a = self.link_style.forward(x, key, style)
        x_b = self.link_context.forward(x, context, context, mask=mask)
        # [B, T, channels]
        x = x_a + x_b
        if mask is not None:
            x = x[..., :1]
        return x


class Refiner(nn.Module):
    """Link attention to retrieve content-specific style for reconstruction.
    """
    def __init__(self,
                 channels: int,
                 contexts: int,
                 styles: int,
                 embeds: int,
                 heads: int,
                 ffn: int,
                 prototypes: int,
                 blocks: int,
                 steps: int,
                 tokens: int,
                 dropout: float = 0.):
        """Initializer.
        Args:
            channels: size of the input channels.
            contexts: size of the context channels.
            styles: size of the style channels.
            embeds: size of the embedding channels.
            heads: the number of the attention heads.
            ffn: size of the FFN hidden channels.
            prototypes: the number of the style vectors.
            blocks: the number of the attention blocks.
            steps: the number of the timesteps.
            tokens: the number of the output tokens.
            dropout: dropout rates for FFN.
        """
        super().__init__()
        self.mapper = nn.Linear(embeds, channels * blocks)
        self.pe = SinusoidalPE(channels)
        # attentions
        self.linkkey = nn.Parameter(torch.randn(1, prototypes, styles))
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                # self attention
                AddNorm(
                    channels,
                    MultiheadAttention(channels, channels, channels, heads)),
                # multiple links
                AuxSequential(
                    AddNorm(
                        channels,
                        MultiLink(channels, contexts, styles, heads)),
                    AddNorm(channels, FeedForward(channels, ffn, dropout)))])
            for _ in range(blocks)])
        # classification head
        self.weight = nn.Parameter(torch.randn(steps, channels, tokens))
        self.bias = nn.Parameter(torch.randn(steps, tokens))

    def forward(self,
                x: torch.Tensor,
                contents: torch.Tensor,
                style: torch.Tensor,
                steps: torch.Tensor,
                embed: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Refine the x with predicting residual signal.
        Args:
            x: [torch.float32; [B, T, channels]], input vectors.
            contents: [torch.float32; [B, S, contexts]], content vectors.
            style: [torch.float32; [B, prototypes, styles]], style vectors.
            steps: [torch.long; [B]], timesteps.
            embed: [torch.float32; [B, embed]], time embeddings.
            mask: [torch.float32; [B, T, S]], mask vector.
        Returns:
            [torch.float32; [B, T, tokens]], logits of residual tokens.
        """
        mask_s = None
        if mask is not None:
            # [B, T]
            mask_s = mask[..., 0]
            # [B, T, T]
            mask_s = mask_s[:, None] * mask_s[..., None]
        # [B, T, channels], positional encoding
        x = x + self.pe.forward(x.shape[1])
        # blocks x [B, channels]
        embeds = self.mapper(embed).chunk(len(self.blocks), dim=-1)
        for (selfattn, linkattn), embed in zip(self.blocks, embeds):
            # [B, T, C], add time-embeddings
            x = x + embed[:, None]
            # [B, T, C]
            x = selfattn(x, x, x, mask=mask_s)
            # [B, T, C]
            x = linkattn(x, self.linkkey, style, contents, mask=mask)
        # [B, T, tokens]
        x = torch.matmul(x, self.weight[steps]) + self.bias[steps, None]
        if mask is not None:
            # [B, T, tokens], masking for FFN.
            x = x * mask[..., :1]
        # [B, T, tokens]
        return x

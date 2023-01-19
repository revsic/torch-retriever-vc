from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.transformer import \
    AddNorm, AuxSequential, FeedForward, LinkAttention, MultiheadAttention


class MultiLink(nn.Module):
    """Attention with multiple kinds.
    """
    def __init__(self,
                 channels: int,
                 contexts: int,
                 styles: int,
                 heads: int,
                 prototypes: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            contexts: size of the context channels.
            styles: size of the style channels.
            heads: size of the FFN hidden channels.
            prototypes: the number of the style vectors.
        """
        super().__init__()
        self.linkkey = nn.Parameter(torch.randn(1, prototypes, styles))
        self.link_style = MultiheadAttention(
            styles, channels, channels // 2, heads // 2)
        self.link_context = MultiheadAttention(
            contexts, channels, channels // 2, heads // 2)
        self.proj = nn.Linear(channels, channels, bias=False)

    def forward(self,
                x: torch.Tensor,
                style: torch.Tensor,
                context: torch.Tensor,
                mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Link the information.
        Args:
            x: [torch.float32; [B, T, channels]], input query.
            style: [torch.float32; [B, prototypes, styles]], style vector.
            context: [torch.float32; [B, S, contexts]], context vector.
            mask: [torch.float32; [B, T, S]], attention mask.
        Returns:
            [torch.float32; [B, T, channels]], attended.
        """
        # [B, T, channels // 2]
        x_a = self.link_style.forward(x, self.linkkey, style)
        x_b = self.link_context.forward(x, context, context, mask=mask)
        # [B, T, channels]
        x = self.proj(torch.cat([x_a, x_b], dim=-1))
        if mask is not None:
            x = x[..., :1]
        return x


class Decoder(nn.Module):
    """Decode the first Encodec tokens from context vectors.
    """
    def __init__(self,
                 contexts: int,
                 kernels: int,
                 styles: int,
                 heads: int,
                 ffn: int,
                 prototypes: int,
                 blocks: int,
                 tokens: int,
                 kappa: float,
                 dropout: float = 0.):
        """Initializer.
        Args:
            contexts: size of the context channels.
            kernels: size of the convolutional kernels, alternatives of PE.
            styles: size of the style channels.
            heads: the number of the attention heads.
            ffn: size of the FFN hidden channels.
            prototypes: the number of the style vectors.
            blocks: the number of the attention blocks.
            tokens: the number of the output tokens.
            kappa: temperature for softmax.
            dropout: dropout rates for FFN.
        """
        super().__init__()
        # instead of positional encoding
        self.preconv = nn.Sequential(
            nn.Conv1d(
                contexts, contexts, kernels,
                padding=kernels // 2, groups=contexts, bias=False),
            nn.ReLU())
        # linker
        self.link = LinkAttention(
            contexts,
            styles,
            heads,
            ffn,
            prototypes,
            blocks,
            dropout)
        # classifier heads
        self.proj = nn.Linear(contexts, contexts)
        self.tokens = nn.Parameter(torch.randn(1, contexts, tokens))
        # alias
        self.kappa = kappa

    def forward(self,
                contents: torch.Tensor,
                style: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Refine the x with predicting residual signal.
        Args:
            contents: [torch.float32; [B, S, contexts]], content vectors.
            style: [torch.float32; [B, prototypes, styles]], style vectors.
            mask: [torch.float32; [B, S]], mask vector.
        Returns:
            [torch.float32; [B, S, tokens]], logits of residual tokens.
        """
        # [B, S, contexts]
        x = self.preconv(contents.transpose(1, 2)).transpose(1, 2)
        # [B, S, contexts]
        x = self.link.forward(x, style, mask=mask)
        # [B, S, tokens]
        x = torch.matmul(
            F.normalize(self.proj(x), p=2, dim=-1),
            F.normalize(self.tokens, p=2, dim=1))
        # temperize
        x = x / self.kappa
        if mask is not None:
            # [B, S, tokens], masking for FFN.
            x = x * mask[..., :1]
        # [B, S, tokens]
        return x


class Refiner(nn.Module):
    """Link attention to retrieve content-specific style for reconstruction.
    """
    def __init__(self,
                 channels: int,
                 kernels: int,
                 contexts: int,
                 styles: int,
                 embeds: int,
                 heads: int,
                 ffn: int,
                 prototypes: int,
                 blocks: int,
                 steps: int,
                 tokens: int,
                 kappa: float,
                 dropout: float = 0.):
        """Initializer.
        Args:
            channels: size of the input channels.
            kernels: size of the convolutional kernels, alternatives of PE.
            contexts: size of the context channels.
            styles: size of the style channels.
            embeds: size of the embedding channels.
            heads: the number of the attention heads.
            ffn: size of the FFN hidden channels.
            prototypes: the number of the style vectors.
            blocks: the number of the attention blocks.
            steps: the number of the timesteps.
            tokens: the number of the output tokens.
            kappa: temperature for softmax.
            dropout: dropout rates for FFN.
        """
        super().__init__()
        self.mapper = nn.Linear(embeds, channels * blocks)
        # instead of positional encoding
        self.preconv = nn.Sequential(
            nn.Conv1d(
                contexts, contexts, kernels,
                padding=kernels // 2, groups=contexts, bias=False),
            nn.ReLU())
        # attentions
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
                        MultiLink(channels, contexts, styles, heads, prototypes)),
                    AddNorm(channels, FeedForward(channels, ffn, dropout)))])
            for _ in range(blocks)])
        # classification head
        self.proj = nn.Linear(channels, channels)
        self.tokens = nn.Parameter(torch.randn(steps, channels, tokens))
        # alias
        self.kappa = kappa

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
        # [B, T, channels], alternatives of positional encoding
        x = self.preconv(x.transpose(1, 2)).transpose(1, 2)
        # blocks x [B, channels]
        embeds = self.mapper(embed).chunk(len(self.blocks))
        for (selfattn, linkattn), embed in zip(self.blocks, embeds):
            # [B, T, C], add time-embeddings
            x = x + embed[:, None]
            # [B, T, C]
            x = selfattn(x, x, x, mask=mask_s)
            # [B, T, C]
            x = linkattn(x, style, contents, mask=mask)
        # [B, T, tokens]
        x = torch.matmul(
            F.normalize(self.proj(x), p=2, dim=-1),
            F.normalize(self.tokens[steps], p=2, dim=1))
        # temperize
        x = x / self.kappa
        if mask is not None:
            # [B, T, tokens], masking for FFN.
            x = x * mask[..., :1]
        # [B, T, tokens]
        return x

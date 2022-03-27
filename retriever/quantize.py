from typing import Tuple

import torch
import torch.nn as nn


class Quantize(nn.Module):
    """Product VQ with gumbel softmax.
    """
    def __init__(self,
                 channels: int,
                 groups: int,
                 vectors: int,
                 temp: float):
        """Initializer.
        Args:
            channels: size of the input channels.
            groups: the number of the product groups.
            vectors: the number of the vectors per groups.
            temp: initial gumbel temperature.
        """
        super(Quantize, self).__init__()
        self.groups, self.vectors = groups, vectors
        self.proj = nn.Linear(channels, groups * vectors)
        self.codebooks = nn.Parameter(
            torch.randn(1, groups, channels // groups, vectors))
        self.register_buffer('temp', torch.tensor(temp, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize inputs.
        Args:
            inputs: [torch.float32; [B, T, C]], input tensor.
        Returns:
            [torch.float32; [B, T, C]], quantized.
            [torch.float32; [B, G, V, T]], logits
        """
        # B, T, _
        bsize, timesteps, _ = inputs.shape
        # [B, G, V, T]
        logits = self.proj(inputs) \
            .view(bsize, timesteps, self.groups, self.vectors) \
            .permute(0, 2, 3, 1)
        gumbels = -torch.log(-torch.log(logits + 1e-5) + 1e-5)
        # [B, G, V, T]
        prob = torch.softmax((logits + gumbels) / self.temp, dim=2)
        # [B, C, T]
        soft = torch.matmul(self.codebooks, prob).view(bsize, -1, timesteps)
        # [B, G, V, T]
        hard = torch.zeros_like(prob).scatter(
            2, prob.argmax(dim=2, keepdim=True), 1.)
        # [B, C, T]
        hard = torch.matmul(self.codebooks, hard).reshape(bsize, -1, timesteps)
        # [B, T, C], [B, G, V, T]
        return (hard - soft.detach() + soft).transpose(1, 2), logits

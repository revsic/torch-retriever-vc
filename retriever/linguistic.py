import torch
import torch.nn as nn

from .attention import AuxSequential, SelfAttention


class LinguisticEncoder(nn.Module):
    """Linguistic encoder.
    """
    def __init__(self,
                 channels: int,
                 kernels: int,
                 hiddens: int,
                 heads: int,
                 ffn: int,
                 blocks: int,
                 dropout: float = 0.):
        """Initializer.
        Args:
            channels: size of the input channels.
            kernels: convolutional kernels.
            hiddens: size of the hidden units.
            heads: the number of the attention heads.
            ffn: size of the FFN hidden channels.
            blocks: the number of the attention blocks.
            dropout: dropout rates for FFN.
        """
        super().__init__()
        self.preconv = nn.Sequential(
            nn.Conv1d(channels, hiddens, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            # instead of positional encoding
            nn.Conv1d(
                hiddens, hiddens, kernels,
                padding=kernels // 2, groups=hiddens, bias=False),
            nn.ReLU())

        self.encoder = AuxSequential(
            SelfAttention(hiddens, heads, ffn, blocks, dropout),
            nn.Linear(hiddens, hiddens))

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Extract the linguistic informations.
        Args:
            inputs: [torch.float32; [B, S, channels]], input tensors.
            mask: [torch.float32; [B, S]], sequential mask.
        Returns:
            [torch.float32; [B, S, hiddens]], extracted.
        """
        # [B, hiddens, S]
        x = self.preconv(inputs.transpose(1, 2))
        # [B, S, hiddens]
        return self.encoder(x.transpose(1, 2), mask)

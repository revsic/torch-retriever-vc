import torch
import torch.nn as nn

from .attention import AuxSequential, LinkAttention


class Decoder(nn.Module):
    """Linguistic encoder.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernels: int,
                 contexts: int,
                 styles: int,
                 heads: int,
                 ffn: int,
                 prototypes: int,
                 blocks: int,
                 detok: int,
                 dropout: float = 0.):
        """Initializer.
        Args:
            in_channels, out_channels: size of the input, output channels.
            kernels: size of the convolutional kernels.
            contexts: size of the context channels.
            styles: size of the style channels.
            heads: the number of the attention heads.
            ffn: size of the FFN hidden channels.
            prototypes: the number of the style vectors.
            blocks: the number of the attention blocks.
            detok: detokenizer FFN hidden channels.
            dropout: dropout rates for FFN.
        """
        super().__init__()
        self.preconv = nn.Sequential(
            nn.Conv1d(in_channels, contexts, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            # instead of positional encoding
            nn.Conv1d(
                contexts, contexts, kernels,
                padding=kernels // 2, groups=contexts, bias=False),
            nn.ReLU())

        self.encoder = AuxSequential(
            LinkAttention(
                contexts,
                styles,
                heads,
                ffn,
                prototypes,
                blocks,
                dropout),
            nn.Linear(contexts, detok),
            nn.ReLU(),
            nn.Linear(detok, out_channels))

    def forward(self,
                inputs: torch.Tensor,
                styles: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Extract the linguistic informations.
        Args:
            inputs: [torch.float32; [B, S, channels]], input tensors.
            styles: [torch.float32; [B, prototypes, styles]], style vectors.
            mask: [torch.float32; [B, S]], sequential mask.
        Returns:
            [torch.float32; [B, S, hiddens]], extracted.
        """
        # [B, hiddens, S]
        x = self.preconv(inputs.transpose(1, 2))
        # [B, S, hiddens]
        return self.encoder(x.transpose(1, 2), styles, mask)

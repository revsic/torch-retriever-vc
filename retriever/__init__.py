from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttention, LinkAttention, SelfAttention
from .config import Config
from .transformer import SinusoidalPE
from .quantize import Quantize


class Retriever(nn.Module):
    """Retreiver for voice conversion.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: model configurations.
        """
        super().__init__()
        self.mel = config.mel
        self.reduction = config.reduction
        self.prenet = nn.Sequential(
            # patch embedding
            nn.Conv1d(
                config.mel, config.channels,
                config.reduction, stride=config.reduction),
            nn.ReLU())

        self.pe = SinusoidalPE(config.channels)

        self.encoder = SelfAttention(
            config.channels, config.heads, config.enc_blocks)

        self.quantize = Quantize(
            config.channels, config.groups, config.vectors, config.temp_max)

        self.retriever = CrossAttention(
            config.channels, config.heads, config.styles, config.ret_blocks)

        self.decoder = LinkAttention(
            config.dec_kernels, config.channels, config.heads,
            config.styles, config.dec_blocks)

        self.proj_out = nn.Linear(config.channels, config.mel * config.reduction)

    def forward(self,
                mel: torch.Tensor,
                mellen: Optional[torch.Tensor] = None,
                refstyle: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Re-generate the spectrogram.
        Args:
            mel: [torch.float32; [B, T x R, mel]], mel-spectrogram.
            mellen: [torch.long; [B]], lengths of the spectrogram.
            refstyle: [torch.float32; [B, S, C]], additional style tokens, if provided.
        Returns:
            [torch.float32; [B, T, mel]], re-sytnehsized spectrogram.
        """
        if mel.shape[1] % self.reduction > 0:
            rest = self.reduction - mel.shape[1] % self.reduction
            # explicit pad
            mel = F.pad(mel, [0, 0, 0, rest])
        else:
            rest = None

        # [B, T, C], patch embedding.
        patch = self.prenet(mel.transpose(1, 2)).transpose(1, 2)
        # [B, T, C], position informing
        patch = patch + self.pe(patch.shape[1])
        if mellen is not None:
            # reduce the lengths
            mellen = torch.ceil(mellen / self.reduction).long()
            # [B, T]
            mask = (
                torch.arange(patch.shape[1], device=mellen.device)[None]
                < mellen[:, None]).to(torch.float32)
            # [B, T, C], premasking
            patch = patch * mask[..., None]
        else:
            mask = None
        # [B, T, C]
        features = self.encoder(patch, mask=mask)
        # [B, T, C], [B, G, V, T]
        contents, logits = self.quantize(features)
        # [B, S, C]
        style = self.retriever(features, mask=mask)
        if refstyle is None:
            # [B, S, C]
            refstyle = style
        # [B, T x R, mel]
        synth = self.proj_out(
                self.decoder(contents, refstyle, mask=mask)
            ).view(contents.shape[0], -1, self.mel)
        if rest is not None:
            # [B, T x R, mel]
            synth = synth[:, :-rest]
        return synth, {
            'contents': contents,
            'style': style,
            'logits': logits}

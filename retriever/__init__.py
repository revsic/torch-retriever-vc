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
                config.mel, config.contexts,
                config.reduction, stride=config.reduction),
            nn.ReLU())

        self.pe = SinusoidalPE(config.contexts)

        self.encoder = SelfAttention(
            config.contexts,    # key, value, query
            config.enc_heads,
            config.enc_ffn,
            config.enc_blocks,
            config.enc_dropout)

        self.quantize = Quantize(
            config.contexts, config.groups, config.vectors, config.temp_max)

        self.retriever = CrossAttention(
            config.contexts,    # key, value
            config.styles,      # query
            config.ret_heads,
            config.ret_ffn,
            config.prototypes,
            config.ret_blocks,
            config.ret_dropout)

        self.decoder = LinkAttention(
            config.dec_kernels,
            config.contexts,
            config.styles,
            config.dec_heads,
            config.dec_ffn,
            config.prototypes,
            config.dec_blocks,
            config.dec_dropout)

        self.proj_out = nn.Sequential(
            nn.Linear(config.contexts, config.detok_ffn),
            nn.ReLU(),
            nn.Linear(config.detok_ffn, config.mel * config.reduction))

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
        # T
        timesteps = mel.shape[1]
        # pad for reduction factor
        if timesteps % self.reduction > 0:
            rest = self.reduction - timesteps % self.reduction
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
        if mask is not None:
            # [B, T, C]
            contents = contents * mask[..., None]
        # [B, S, C]
        style = self.retriever(features, mask=mask)
        if refstyle is None:
            # [B, S, C]
            refstyle = style
        # [B, T x R, mel]
        synth = self.proj_out(
                self.decoder(contents, refstyle, mask=mask)
            ).view(contents.shape[0], -1, self.mel)
        if mask is not None:
            # [B, T x R, mel]
            synth = synth * mask[..., None]
        # unpad
        if rest is not None:
            # [B, T x R, mel]
            synth = synth[:, :-rest]
        return synth, {
            'contents': contents,
            'style': style,
            'logits': logits}

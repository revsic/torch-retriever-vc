from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttention, LinkAttention, SelfAttention
from .config import Config
from .transformer import SinusoidalPE, SequentialWrapper
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

        self.cpcpred = nn.GRU(
            config.contexts, config.contexts,
            num_layers=config.cpc_layers, batch_first=True)
        self.cpc_proj = nn.ModuleList([
            nn.Linear(config.contexts, config.contexts)
            for _ in range(config.cpc_steps)])

        self.retriever = CrossAttention(
            config.contexts,    # key, value
            config.styles,      # query
            config.ret_heads,
            config.ret_ffn,
            config.prototypes,
            config.ret_blocks,
            config.ret_dropout)

        self.decoder = SequentialWrapper(
            LinkAttention(
                config.dec_kernels,
                config.contexts,
                config.styles,
                config.dec_heads,
                config.dec_ffn,
                config.prototypes,
                config.dec_blocks,
                config.dec_dropout),
            nn.Sequential(
                nn.Linear(config.contexts, config.detok_ffn),
                nn.ReLU(),
                nn.Linear(config.detok_ffn, config.mel * config.reduction)))

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
        # T x R
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
        # [B, T, R x mel]
        synth = self.decoder(contents, refstyle, mask=mask)
        if mask is not None:
            # [B, T, R x mel]
            synth.masked_fill_(~mask[..., None].to(torch.bool), np.log(1e-5))
        # [B, T x R, mel]
        synth = synth.view(synth.shape[0], -1, self.mel)
        # unpad
        if rest is not None:
            # [B, T x R, mel]
            synth = synth[:, :-rest]
        return synth, {
            'features': features,
            'contents': contents,
            'style': style,
            'logits': logits}

    def save(self, path: str, optim: Optional[torch.optim.Optimizer] = None):
        """Save the models.
        Args:
            path: path to the checkpoint.
            optim: optimizer, if provided.
        """
        dump = {'model': self.state_dict()}
        if optim is not None:
            dump['optim'] = optim.state_dict()
        torch.save(dump, path)

    def load(self, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        self.load_state_dict(states['model'])
        if optim is not None:
            optim.load_state_dict(states['optim'])

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttention, LinkAttention
from .attention.transformer import AddNorm, FeedForward
from .config import Config
from .linguistic import LinguisticEncoder
from .wav2vec2 import Wav2Vec2Wrapper


class Retriever(nn.Module):
    """Retreiver for voice conversion.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: model configurations.
        """
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Wrapper(
            config.w2v2_name,
            config.sr)

        # LinguisticEncoder
        self.linguistic = LinguisticEncoder(
            self.wav2vec2.channels,
            config.ling_hiddens,
            config.ling_heads,
            config.ling_ffn,
            config.ling_blocks,
            config.ling_dropout)

        self.retriever = CrossAttention(
            self.wav2vec2.channels,
            config.styles,
            config.ret_heads,
            config.ret_ffn,
            config.prototypes,
            config.ret_blocks,
            config.ret_dropout)

        self.proj_context = nn.Sequential(
            nn.Linear(config.ling_hiddens, config.contexts),
            AddNorm(
                config.contexts,
                FeedForward(config.contexts, config.dec_ffn)))

        self.decoder = LinkAttention(
            config.contexts,
            config.styles,
            config.dec_heads,
            config.dec_ffn,
            config.prototypes,
            config.dec_blocks,
            config.dec_dropout)
        
        self.detok = nn.Sequential(
            nn.Linear(config.contexts, config.dec_ffn),
            nn.ReLU(),
            nn.Linear(config.dec_ffn, config.mel))

    def forward(self,
                audio: torch.Tensor,
                audlen: Optional[torch.Tensor] = None,
                refstyle: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize the codebook.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
            audlen: [torch.long; [B]], lengths of the audio.
            refstyle: [torch.float32; [B, prototypes, styles]], style tokens, use it if provided.
        Returns:
            [torch.long; [B, T / hop, mel]], sytnehsized audio.
            [torch.float32; [B, prototypes, styles]], style of the given audio.
        """
        device = audio.device
        # B, T
        bsize, timesteps = audio.shape
        # [B], placeholder
        audlen = audlen or torch.full((bsize,), timesteps, device=device)
        # 2 x [B, L, w2v2_channels], [B, L]
        spk, ling, mask = self.wav2vec2.forward(audio, audlen)
        # [B, L, ling_hiddens]
        ling = self.linguistic.forward(ling, mask)
        # [B, prototypes, styles]
        style = self.retriever.forward(spk, mask)

        # [B]
        mellen = (audlen / self.config.hop).ceil().long()
        # S
        mlen = mellen.amax().item()
        # [B, S]
        mask_c = (
            torch.arange(mlen, device=device)[None]
            < mellen[:, None]).float()

        # [B, L, contexts]
        states = self.proj_context(ling)
        # [B, contexts, S], length-aware interpolation
        states = torch.cat([
            F.pad(
                F.interpolate(c[None, :, :l.item()], size=s.item(), mode='nearest'),
                [0, mlen - s.item()])
            for c, s, l in zip(
                states.transpose(1, 2),
                mellen,
                mask.sum(dim=-1).long())], dim=0)
        # [B, prototypes, styles], selecting style
        if refstyle is None:
            refstyle = style
        # [B, S, contexts]
        states = self.decoder.forward(states.transpose(1, 2), refstyle, mask_c)
        # [B, S, mel]
        mel = self.detok(states) * mask_c[..., None]
        return mel, style

    def save(self, path: str, optim: Optional[torch.optim.Optimizer] = None):
        """Save the models.
        Args:
            path: path to the checkpoint.
            optim: optimizer, if provided.
        """
        dump = {'model': self.state_dict(), 'config': vars(self.config)}
        if optim is not None:
            dump['optim'] = optim.state_dict()
        torch.save(dump, path)

    def load_(self, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints inplace.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        self.load_state_dict(states['model'], strict=False)
        if optim is not None:
            optim.load_state_dict(states['optim'])

    @classmethod
    def load(cls, states: Dict[str, Any], optim: Optional[torch.optim.Optimizer] = None):
        """Load from checkpoints.
        Args:
            states: state dict.
            optim: optimizer, if provided.
        """
        config = Config()
        for key, val in states['config'].items():
            if not hasattr(config, key):
                import warnings
                warnings.warn(f'unidentified key {key}')
                continue
            setattr(config, key, val)
        # construct
        retv = cls(config)
        retv.load_(states, optim)
        return retv

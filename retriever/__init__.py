from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AuxSequential, CrossAttention, LinkAttention, SinusoidalPE
from .config import Config
from .decoder import Refiner
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

        self.proj_fst = nn.Linear(config.ling_hiddens, config.contexts)
        self.dec_fst = AuxSequential(
            LinkAttention(
                config.contexts,
                config.styles,
                config.fst_heads,
                config.fst_ffn,
                config.prototypes,
                config.fst_blocks,
                config.fst_dropout),
            nn.Linear(config.contexts, config.encodecs))

        self.steps = nn.Sequential(
            SinusoidalPE(config.pe, config.timesteps - 1),
            nn.Linear(config.pe, config.embeds),
            nn.ReLU(),
            nn.Linear(config.embeds, config.embeds),
            nn.ReLU())
        
        self.codebooks = nn.ModuleList([
            nn.Embedding(config.encodecs, config.contexts)
            for _ in range(config.timesteps - 1)])

        self.dec_rst = Refiner(
            config.contexts,
            config.ling_hiddens,
            config.styles,
            config.embeds,
            config.rst_heads,
            config.rst_ffn,
            config.prototypes,
            config.rst_blocks,
            config.timesteps - 1,
            config.encodecs,
            config.rst_dropout)

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
            [torch.long; [B, timesteps, S]], sytnehsized audio codecs.
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
        # [B, prototypes, tyles]
        style = self.retriever.forward(spk, mask)

        # [B]
        codelen = (audlen / Config.ENCODEC_STRIDES).ceil().long()
        # S
        clen = codelen.amax().item()
        # [B, S]
        mask_c = (
            torch.arange(clen, device=device)[None]
            < codelen[:, None]).to(torch.float32)
        # [B, contexts, S], length-aware interpolation
        ling_fst = torch.cat([
            F.pad(
                F.interpolate(l[None, :, :v.item()], size=s.item(), mode='linear'),
                [0, clen - s.item()])
            for l, v, s in zip(
                self.proj_fst(ling).transpose(1, 2),
                mask.sum(dim=-1).long(),
                codelen)], dim=0)

        # [B, prototypes, styles], selecting style
        refstyle = refstyle or style
        # [B, S, tokens]
        logits = self.dec_fst.forward(ling_fst.transpose(1, 2), refstyle, mask=mask_c)
        # [B x S, tokens]
        prob = torch.softmax(logits, dim=-1).view(-1, self.config.encodecs)
        # [B, S], sampling
        code = torch.multinomial(prob, 1).view(bsize, -1)
        # timesteps x [B, S], [B, S, contexts]
        codes, cumsum = [code], torch.zeros(bsize, clen, self.config.contexts, device=device)
        # [B, S, L]
        mask = mask_c[..., None] * mask[:, None]
        # [timesteps - 1, embeds]
        embeds = self.steps(self.config.timesteps - 1)
        # [embeds], [encodecs, contexts]
        for i, (pe, codebook) in enumerate(zip(embeds, self.codebooks)):
            # [B, S, contexts]
            cumsum = cumsum + codebook(code) * mask_c[..., None]
            # [1]
            steps = torch.tensor([i], device=device)
            # [B, S, tokens]
            logits = self.dec_rst.forward(
                cumsum, ling, refstyle, steps, pe[None], mask=mask)
            # [B, S], greedy search
            code = logits.argmax(dim=-1)
            codes.append(code)
        # [B, timesteps, S]
        return torch.stack(codes, dim=1) * mask_c[:, None].long(), style

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
        self.load_state_dict(states['model'])
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

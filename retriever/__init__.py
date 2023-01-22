from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossAttention, SinusoidalPE
from .config import Config
from .decoder import Decoder
from .encodec import EncodecWrapper
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
        self.encodec = EncodecWrapper(config.sr)
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

        self.start = nn.Parameter(torch.randn(config.contexts))
        self.dec_fst = Decoder(
            config.contexts,
            config.ling_hiddens,
            config.styles,
            config.embeds,
            config.fst_heads,
            config.fst_ffn,
            config.prototypes,
            config.fst_blocks,
            1,
            config.encodecs,
            causal=True,
            dropout=config.fst_dropout)

        self.steps = nn.Sequential(
            SinusoidalPE(config.pe, config.timesteps - 1),
            nn.Linear(config.pe, config.embeds),
            nn.ReLU(),
            nn.Linear(config.embeds, config.embeds),
            nn.ReLU())

        self.proj_code = nn.Linear(Config.ENCODEC_DIM, config.contexts)

        self.dec_rst = Decoder(
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
            dropout=config.rst_dropout)

    @property
    def quantizers(self):
        return self.encodec.model.quantizer.vq.layers

    def embedding(self, vq: 'VectorQuantization', code: torch.Tensor) -> torch.Tensor:
        """Embed with pretrained encodec vector quantizer.
        Args:
            vq: encodec.quantization.core_vq.VectorQuantization
            code: [torch.long; [...]], code sequence.
        Returns:
            [torch.float32; [..., contexts]], projected embedding.
        """
        # [..., Config.ENCODEC_DIM]
        embed = F.embedding(code, vq.codebook.detach())
        # [..., contexts]
        return self.proj_code(embed)

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
            [torch.long; [B, T]], sytnehsized audio.
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

        # [B, prototypes, styles], selecting style
        refstyle = refstyle or style
        # S x [B, 1], [1, 1, contexts], start token.
        codes, cumul = [], self.start[None, None]
        # split code book
        quant_fst, *quant_rst = self.quantizers
        # [1, E]
        null = torch.zeros(1, self.config.embeds, device=device)
        for _ in range(clen):
            # [B, _, tokens]
            logits = self.dec_fst.forward(cumul, ling, style, 0, null)
            # [B, tokens]
            prob = logits[:, -1].softmax(dim=-1)
            # [B, 1], sampling
            code = torch.multinomial(prob, 1)
            codes.append(code)
            # [B, _ + 1, contexts]
            cumul = torch.cat([cumul, self.embedding(quant_fst, code)], dim=1)
        # [B, S]
        code = torch.cat(codes, dim=-1)
        # timesteps x [B, S], [B, S, contexts]
        codes, cumul = [code], cumul[:, 1:] * mask_c[..., None]
        # [B, S, L]
        mask = mask_c[..., None] * mask[:, None]
        # [timesteps - 1, embeds]
        embeds = self.steps(self.config.timesteps - 1)
        # [embeds], [encodecs, contexts]
        for i, (embed, quant) in enumerate(zip(embeds, quant_rst + [None])):
            # [B, S, tokens]
            logits = self.dec_rst.forward(
                cumul, ling, refstyle, i, embed[None], mask=mask)
            # [B, S], greedy search
            code = logits.argmax(dim=-1)
            codes.append(code)
            # [B, S, contexts], cumulate
            cumul = cumul + self.embedding(quant, code) * mask_c[..., None] 
        # [B, timesteps, S]
        codes = torch.stack(codes, dim=1) * mask_c[:, None].long()
        # [B, T']
        wav = self.encodec.decode(codes)
        # masking
        mask = (
            torch.arange(timesteps, device=device)[None]
            < audlen[:, None]).to(device)
        # [B, T], [B, prototypes, styles]
        return wav[:, :timesteps] * mask, style

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

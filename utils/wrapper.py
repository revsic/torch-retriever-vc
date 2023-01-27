from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from config import Config
from retriever import Retriever

from .augment import Augment


class TrainingWrapper:
    """Training wrapper.
    """
    def __init__(self, model: Retriever, config: Config, device: torch.device):
        """Initializer.
        Args:
            model: retriever.
            config: training configurations.
            device: torch device.
        """
        self.model = model
        self.config = config
        self.device = device
        self.aug = Augment(config).to(device)
        # [win] auxiliary tensors
        self.window = torch.hann_window(config.model.win, device=device)
        # [mel, fft // 2 + 1]
        self.fbank = torchaudio.functional.melscale_fbanks(
            config.model.fft // 2 + 1,
            config.model.fmin,
            config.model.fmax,
            config.model.mel,
            config.model.sr,
            norm='slaney', mel_scale='slaney').to(device).T
        # alias
        self.seglen = self.config.train.seglen
        self.lambda_cont = self.config.train.cont_start

    def segment(self,
                seq: torch.Tensor,
                len_: Optional[torch.Tensor] = None,
                start: Optional[torch.Tensor] = None,
                seglen: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment the spectrogram and audio into fixed sized array.
        Args:
            seq: [np.float32; [B, ..., T]], sequence tensor.
            len_: [torch.long; [B]], sequence lengths.
            start: [torch.long; [B]], start position, randomize start position if not provided.
            seglen: segment length, use `self.seglen` if not provided.
        Returns:
            segmented sequence and start position
        """
        # placeholder
        seglen = seglen or self.seglen
        if start is None:
            assert len_ is not None, 'either `len_` or `start` should be passed'
            # B
            bsize, = len_.shape
            # [B]
            start = torch.rand(bsize, device=self.device) * (len_ - seglen).clamp_min(0)
            start = start.long()
        # [B, seglen]
        seg = torch.stack([
            F.pad(q[..., s:s + seglen], [0, max(seglen - q.shape[-1], 0)])
            for q, s in zip(seq, start.tolist())], dim=0)
        return seg, start

    def compute_loss(self, speeches: torch.Tensor, lengths: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, float], Dict[str, np.ndarray]]:
        """Compute the loss.
        Args:
            seg: [torch.float32; [B, T]], segmented speech.
        Returns:
            loss and dictionaries.
        """
        with torch.no_grad():
            # [B, fft // 2 + 1, T // hop]
            spec = torch.stft(
                speeches,
                self.config.model.fft,
                self.config.model.hop,
                self.config.model.win,
                self.window,
                return_complex=True).abs()
            # [B, mel, T // hop]
            mel = torch.matmul(self.fbank, spec).clamp_min(1e-5).log()
            # T // hop
            _, _, mlen = mel.shape
            # [B]
            mellen = (lengths / self.config.model.hop).ceil().long().clamp_max(mlen)
            # [B, mel, seglen // hop]
            hop = self.config.model.hop
            mel, start = self.segment(mel, mellen, seglen=self.seglen // hop)
            # [B, seglen]
            seg, _ = self.segment(speeches, start=start * hop)
            # [B, L, w2v2_channels]
            spk, _, _ = self.model.wav2vec2.forward(seg)

            # [B x 2, seglen]
            aug = self.aug.augment(seg.repeat(2, 1))
            # [B x 2, L, w2v2_channels]
            _, ling, _ = self.model.wav2vec2.forward(aug)
            # [B, L, w2v2_channels]
            ling1, ling2, = ling.chunk(2, dim=0)
        
        # [B, L, ling_hiddens]
        ling = self.model.linguistic.forward(ling1)
        # [B, prototypes, styles]
        style = self.model.retriever.forward(spk)

        # [B, L, contexts]
        states = self.model.proj_context(ling)
        # [B, contexts, S]
        states = F.interpolate(states.transpose(1, 2), size=self.seglen // hop, mode='nearest')
        # [B, S, contexts]
        states = self.model.decoder.forward(states.transpose(1, 2), style)
        # [B, S, mel]
        pred = self.model.detok(states)

        # []
        rctor = F.l1_loss(mel.transpose(1, 2), pred)

        # [2, B, L, ling_hiddens]
        ling_ex = torch.stack([
            ling,
            self.model.linguistic.forward(ling2)], dim=0)
        # L
        num_tokens = ling_ex.shape[2]
        # alias
        conf_t = self.config.train
        n_adj, n_cand, kappa = conf_t.num_adj, conf_t.num_cand, conf_t.kappa
        # [B, L], positive
        pos = ling_ex.prod(dim=0).sum(dim=-1) / kappa
        # [2, B, L, L]
        confusion = torch.matmul(ling_ex, ling_ex.transpose(2, 3)) / kappa
        # [L]
        placeholder = torch.zeros(num_tokens, device=self.device)
        # [L, L]
        mask = torch.stack([
            placeholder.scatter(
                0,
                (
                    torch.randperm(num_tokens - n_adj, device=self.device)[:n_cand]
                    + i + n_adj // 2 + 1) % num_tokens,
                1.)
            for i in range(num_tokens)])
        # include self
        mask = mask + torch.eye(num_tokens, device=self.device)
        # [2, B, L, L(sum = candidates)], negative case
        masked = confusion.masked_fill(~mask.to(torch.bool), -np.inf)
        # [2, B, L], negative case
        neg = torch.logsumexp(masked, dim=-1)
        # []
        cont_loss = -torch.logsumexp(pos - neg, dim=-1).sum(dim=0).mean()

        # metric purpose
        metric_pos = pos.mean().item() * kappa
        metric_neg = ((confusion * mask).sum(dim=-1) / n_cand).mean().item() * kappa

        # []
        loss = rctor + self.lambda_cont * cont_loss
        losses = {
            'loss/loss': loss.item(),
            'loss/rctor': rctor.item(),
            'loss/cont': cont_loss.item(),
            'metric/neg': metric_neg,
            'metric/pos': metric_pos,
            'common/warmup': self.lambda_cont}
        return loss, losses, {
            'seg': seg.cpu().detach().numpy(),
            'mel': mel.transpose(1, 2).cpu().detach().numpy(),
            'rctor': pred.cpu().detach().numpy()}

    def update_warmup(self):
        """Update the content loss weights.
        """
        conf_t = self.config.train
        self.lambda_cont = min(self.lambda_cont + conf_t.cont_start, conf_t.cont_end)

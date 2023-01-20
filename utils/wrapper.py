from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from retriever import Retriever

from .augment import Augment
from .encodec import EncodecWrapper


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
        self.encodec = EncodecWrapper(config.model.sr).to(device)
        # alias
        self.seglen = self.config.train.seglen

    def random_segment(self, bunch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Segment the spectrogram and audio into fixed sized array.
        Args:
            bunch: input tensors.
                speeches: [np.float32; [B, T]], speeches.
                lengths: [np.long; [B]], speech lengths.
        Returns:
            randomly segmented spectrogram and audios.
        """
        # [B, T], [B]
        speeches, lengths = bunch
        # B
        bsize, = lengths.shape
        # [B]
        start = torch.rand(bsize, device=self.device) * (lengths - self.seglen).clamp_min(0)
        # [B, seglen]
        seg = torch.stack([
            F.pad(q[s:s + self.seglen], [0, max(self.seglen - q.shape[-1], 0)])
            for q, s in zip(speeches, start.long())], dim=0)
        return lengths.clamp_max(self.seglen), seg

    def compute_loss(self, seg: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss.
        Args:
            seg: [torch.float32; [B, T]], segmented speech.
        Returns:
            loss and dictionaries.
        """
        with torch.no_grad():
            # [B, num_q, S]
            codebook = self.encodec.forward(seg)
            # [B, S]
            aug1 = self.aug.augment(seg)
            aug2 = self.aug.augment(seg)
            # 2 x [B, L, w2v2_channels], [B, L]
            spk, ling, _ = self.model.wav2vec2.forward(aug1)
            # [B, L, w2v2_channels]
            _, ling2, _ = self.model.wav2vec2.forward(aug2)
        # [B, L, ling_hiddens]
        ling = self.model.linguistic.forward(ling)
        # [B, prototypes, styles]
        style = self.model.retriever.forward(spk)

        # B, S
        bsize, _, clen = codebook.shape
        # [B, contexts, S]
        ling_fst = F.interpolate(
            self.model.proj_fst(ling).transpose(1, 2), size=clen, mode='linear')
        # [B, S, tokens]
        logits = self.model.dec_fst.forward(ling_fst.transpose(1, 2), style)
        # []
        ce_fst = F.cross_entropy(
            torch.softmax(logits.transpose(1, 2), dim=1),
            codebook[:, 0])
        # metric purpose
        metric_fst = (logits.argmax(dim=-1) == codebook[:, 0]).float().mean().item()

        # alias
        timesteps = self.config.model.timesteps
        # [B], [0, timesteps - 1)
        steps = (torch.rand(bsize, device=self.device) * (timesteps - 1)).long()
        # [timesteps - 1, embeds]
        embeds = self.model.steps(timesteps - 1)

        # [B, S, contexts]
        codes = torch.stack([
            torch.stack([
                self.model.codebooks[j](code[j])
                for j in range(i + 1)], dim=0).sum(dim=0)
            for code, i in zip(codebook, steps.tolist())], dim=0)
        # [B, S, tokens]
        logits = self.model.dec_rst.forward(
            codes, ling, style, steps, embeds[steps])
        # [B]
        aranger = torch.arange(bsize, device=self.device)
        # []
        ce_rst = F.cross_entropy(
            torch.softmax(logits.transpose(1, 2), dim=1),
            codebook[aranger, steps + 1])
        # metric purpose
        with torch.no_grad():
            # [B]
            acc_rst = (logits.argmax(dim=-1) == codebook[aranger, steps + 1]).float().mean(dim=-1)
            # []
            metric_rst = acc_rst.mean().item()
            # step-wise aggregation
            steps = steps.tolist()
            agg = {s: [] for s in steps}
            for s, a in zip(steps, acc_rst.tolist()):
                agg[s].append(a)
            metric_agg = {s + 1: sum(a) / max(len(a), 1) for s, a in agg.items()}

        # [2, B, L, ling_hiddens]
        ling_ex = torch.stack([
            ling,
            self.model.linguistic.forward(ling2)], dim=0)
        # normalize for cossim
        ling_ex = F.normalize(ling_ex, dim=-1)
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
        metric_pos = pos.mean().item()
        metric_neg = ((confusion * mask).sum(dim=-1) / n_cand).mean().item()

        # []
        loss = ce_fst + ce_rst + conf_t.lambda_cont * cont_loss
        losses = {
            'loss/loss': loss.item(),
            'loss/ce-fst': ce_fst.item(),
            'loss/ce-rst': ce_rst.item(),
            'loss/cont': cont_loss.item(),
            'metric/neg': metric_neg,
            'metric/pos': metric_pos,
            'metric/fst': metric_fst,
            'metric/rst': metric_rst}
        losses.update({f'metric-aux/step{i}': acc for i, acc in metric_agg.items()})
        return loss, losses

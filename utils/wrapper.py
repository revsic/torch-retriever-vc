from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from retriever import Retriever


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
        # alias
        self.silence = np.log(config.data.eps)
        self.seglen = config.train.seglen
        self.vectors = config.model.vectors
        self.gamma = np.log(config.model.vectors)
        self.pos = config.model.cpc_pos
        self.neg = config.model.cpc_neg

    def random_segment(self, bunch: List[np.ndarray]) -> np.array:
        """Segment the spectrogram and audio into fixed sized array.
        Args:
            bunch: input tensors.
                mel: [np.float32; [B, T, mel]], mel-spectrogram.
                mellen: [np.long; [B]], spectrogram lengths.
        Returns:
            randomly segmented spectrogram.
        """
        # [B, T, mel], [B]
        _, _, mel, _, mellen = bunch
        # [B], min clipping
        start = np.random.randint(np.maximum(1, mellen - self.seglen))
        # B x [seglen, mel]
        seg = []
        for m, s in zip(mel, start):
            if len(m) < self.seglen:
                m = np.pad(m, [[0, self.seglen - len(m)], [0, 0]],
                           constant_values=self.silence)
            seg.append(m[s:s + self.seglen])
        # [B, seglen, mel]
        return np.array(seg)

    def compute_loss(self, mel: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute the loss.
        Args:
            mel: [torch.float32; [B, seglen, mel]], segmented spectrogram.
        Returns:
            loss and dictionaries.
        """
        # [B, seglen, mel], _
        synth, aux = self.model(mel)

        ## 1. reconstruction
        rec = F.l1_loss(mel, synth)

        ## 2. VQ perplexity
        # [G, V], mean across batch
        prob = torch.softmax(aux['logits'], dim=2).mean(dim=[0, 3])
        # [G]
        perplexity = (prob * torch.log(prob + 1e-5)).sum(dim=1).exp() / self.vectors
        # []
        perplexity = perplexity.mean()

        ## 3. Masked lm
        def randmask(size: int, samples: int, device: torch.device) -> torch.Tensor:
            return torch.zeros(size, device=device).scatter(
                0, torch.randperm(size, device=device)[:samples], 1.)
        # B, T, _
        bsize, timesteps, _ = aux['features'].shape
        # [T]
        mask = randmask(timesteps, self.pos, aux['features'].device)
        # [B, T, C]
        pred = self.model.cpcpred(
            aux['features'] + (
                self.model.maskembed - aux['features']) * mask[None, :, None])
        # [B, T, T], cosine similarity
        logit = torch.matmul(
            F.normalize(pred, dim=-1),
            F.normalize(aux['contents'], dim=-1).transpose(1, 2))
        # for numerical stability
        # , since range of cossim is [-1, 1], logit.max() == 1.
        ratio = torch.exp(logit - 1.)

        # [T, T]
        pos_mask = mask[:, None] * mask[None]
        # [B, T]
        log_pos = torch.log((ratio * pos_mask[None]).sum(dim=-1) + 1e-7)
        
        # [T, T]
        neg_mask = randmask(
            timesteps, self.neg, ratio.device)[None].repeat(timesteps, 1) * (1 - pos_mask)
        # [B, T]
        neg = (ratio * neg_mask[None]).sum(dim=-1)

        
        # [], range [1, B - 1]
        start = np.random.randint(bsize - 1) + 1
        # [B], for shuffling
        indices = (np.arange(bsize) + start) % bsize
        # [B, T, T], cross-batch cossim
        cross_logit = torch.matmul(
            F.normalize(pred, dim=-1),
            F.normalize(aux['contents'][indices], dim=-1).transpose(1, 2))
        # for numerical stability
        cross_ratio = torch.exp(cross_logit - 1.)
        # [B, T]
        log_neg = torch.log(neg + (cross_ratio * neg_mask[None]).sum(dim=-1) + 1e-7)

        # []
        infonce = -(log_pos - log_neg).mean()

        # []
        loss = self.config.train.lambda_rec * rec + \
            self.config.train.lambda_vq * perplexity + \
            self.config.train.lambda_sc * infonce
        losses = {
            'loss': loss.item(),
            'rec': rec.item(), 'vq': perplexity.item(), 'cpc': infonce.item()}
        return loss, losses, {'synth': synth.cpu().detach().numpy()}

    def update_gumbel_temp(self):
        """Update gumbol softmax temperature.
        """
        # update temperature
        temp = max(
            self.model.quantize.temp.item() * self.config.model.temp_factor,
            self.config.model.temp_min)
        # re-register buffer
        self.model.quantize.register_buffer(
            'temp', torch.tensor(temp, dtype=torch.float32, device=self.device))

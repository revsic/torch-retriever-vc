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
        self.sample = config.model.cpc_sample

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

        ## 3. Structural constraint
        # [B, V, T], sample first group
        groups = aux['logits'][:, 0]
        # [B, T]
        labels = groups.argmax(dim=1)
        # [B, T - 1]
        ce = F.cross_entropy(groups[..., :-1], labels[..., 1:], reduction='none') \
            + F.cross_entropy(groups[..., 1:], labels[..., :-1], reduction='none')
        # [B]
        ce = ce.mean(dim=-1)
        # [B]
        structural = self.gamma * torch.tanh(ce / self.gamma)
        # []
        structural = structural.mean()

        ## 4. Masked lm
        # T
        timesteps = aux['features'].shape[1]
        # [T]
        mask = torch.zeros(timesteps, device=self.device).scatter(
            0, torch.randperm(timesteps, device=self.device)[:self.sample], 1.)
        # [B, T, C]
        pred = self.model.cpcpred(
            aux['features'] + (
                self.model.maskembed - aux['features']) * mask[None, :, None])
        # [B, T, T], cosine similarity
        logit = torch.matmul(
            F.normalize(pred, dim=-1),
            F.normalize(aux['contents'], dim=-1).transpose(1, 2))
        # for numerical stability, torch.exp(logit - logit.max())
        ratio = torch.exp(logit - 1.)

        # [T, T]
        pos_mask = torch.eye(timesteps, device=self.device) * mask
        # [B, T]
        log_pos = torch.log((ratio * pos_mask[None]).sum(dim=-1) + 1e-7)

        # [T, T], exclusive candidates
        neg_mask = mask[None] * mask[:, None] * (1 - pos_mask)
        # [B, T]
        log_neg = torch.log((ratio * neg_mask[None]).sum(dim=-1) + 1e-7)

        # []
        infonce = -(log_pos - log_neg).mean()

        # []
        loss = self.config.train.lambda_rec * rec + \
            self.config.train.lambda_vq * perplexity + \
            self.config.train.lambda_sc * (structural + infonce)
        losses = {
            'loss': loss.item(),
            'rec': rec.item(), 'vq': perplexity.item(),
            'sc': structural.item(), 'cpc': infonce.item()}
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

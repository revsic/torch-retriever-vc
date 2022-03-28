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
        self.seglen = self.config.train.seglen
        self.gamma = np.log(self.config.model.vectors)

    def wrap(self, bunch: List[np.ndarray]) -> List[torch.Tensor]:
        """Wrap the array to torch tensor.
        Args:
            bunch: input tensors.
        Returns:
            wrapped.
        """
        return [torch.tensor(array, device=self.device) for array in bunch]

    def random_segment(self, bunch: List[np.ndarray]) -> List[np.ndarray]:
        """Segment the spectrogram and audio into fixed sized array.
        Args:
            bunch: input tensors.
                mel: [np.float32; [B, T, mel]], mel-spectrogram.
                mellen: [np.long; [B]], spectrogram lengths.
        Returns:
            randomly segmented spectrogram.
        """
        # [B, T, mel], [B]
        mel, _, mellen, _ = bunch
        # [B], min clipping
        start = np.random.randint(np.maximum(1, mellen - self.seglen))
        # B x [seglen, mel]
        seg = []
        for m, s in zip(mel, start):
            if len(m) < self.seglen:
                m = np.pad(m, [[0, self.seglen - len(m)], [0, 0]])
            seg.append(m[s:s + self.seglen])
        # [B, seglen, mel], [B], fixed length segment
        return np.array(seg), np.minimum(mellen, self.seglen)

    def compute_loss(self, mel: torch.Tensor, mellen: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute the loss.
        Args:
            mel: [torch.float32; [B, seglen, mel]], segmented spectrogram.
            mellen: [torch.float32; [B]], segment length.
        Returns:
            loss and dictionaries.
        """
        # [B, seglen, mel], _
        synth, aux = self.model(mel, mellen)

        ## 1. reconstruction
        rec = F.l1_loss(mel, synth)

        ## 2. VQ perplexity
        # [B, G, V, T]
        prob = torch.softmax(aux['logits'], dim=2)
        # [B, G, T]
        perplexity = (prob * torch.log(prob + 1e-5)).sum(dim=2).exp() / self.config.model.vectors
        # []
        perplexity = perplexity.mean()

        ## 3. Structural constraint
        # [B, V, T], sample first group
        groups = prob[:, 0]
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

        # []
        loss = self.config.train.lambda_rec * rec + \
            self.config.train.lambda_vq * perplexity + \
            self.config.train.lambda_sc * structural
        losses = {
            'loss': loss.item(),
            'rec': rec.item(), 'vq': perplexity.item(), 'sc': structural.item()}
        return loss, losses, {'synth': synth}

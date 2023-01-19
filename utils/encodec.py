import sys
import os
# __file__ == ./utils/encodec.py
ENCODEC_PATH = os.path.join(os.path.dirname(__file__), '..', 'encodec')
sys.path.append(ENCODEC_PATH)

from encodec import EncodecModel
sys.path.pop()

from typing import Optional

import torch
import torchaudio


class EncodecWrapper:
    """Encodec Wrapper
    """
    # hardcoded
    STRIDES = 320

    def __init__(self, sr: int, device: Optional[torch.device] = None):
        """Initializer.
        Args:
            device: target computing device.
        """
        device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        # return evalution mode
        self.model = EncodecModel.encodec_model_24khz()
        self.model.to(device)

        self.resampler = torchaudio.transforms.Resample(sr, self.model.sample_rate)
        self.resampler.to(device)
        # alias
        self.num_q = self.model.quantizer.n_q
        self.sr = self.model.sample_rate

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Inference the codebook from wav.
        Args:
            audio: [torch.float32; [B, T]], audio sequence, [-1, 1]-ranged.
        Returns:
            [torch.long; [B, num_q, S / STRIDES]], codebook,
                where S = T / `sr` x `model.sample_rate`
        """
        # [B, 1(=mono), S]
        audio = self.resampler(audio)[:, None]
        # SUBBATCH x [([B, n_q, L], _)]
        frames = self.model.encode(audio)
        # [B, n_q, S / strides]
        return torch.cat([encoded for encoded, _ in frames], dim=-1)

    @torch.no_grad()
    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """Recover the codebook.
        Args:
            codebook: [torch.long; [B, n_q, S / STRIDES]],
                codebook, [-1, 1]-ranged, `sr`-sampling rated
        Returns:
            [torch.long; [B, S]], recovered audio.
        """
        # [B, 1(=mono), S]
        audio = self.model.decode([(code, None)])
        # [B, S]
        return audio.squeeze(dim=1)

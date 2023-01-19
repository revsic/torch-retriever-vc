from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


def phase_vocoder(stft: torch.Tensor, rate: torch.Tensor, hop: int):
    """Speed up in time without modifying pitch.
    Args:
        stft: [torch.complex64; [B, fft // 2 + 1, T / hop]], fourier features.
        rate: [torch.float32; [B]], speed rate.
    """
    # fft // 2 + 1
    _, num_freq, timesteps  = stft.shape
    # [num_freq]
    phase_adv = torch.linspace(0, np.pi * hop, num_freq, device=stft.device)
    # B x [L(=T / hop / rate)]
    steps = [
        torch.arange(0, timesteps, r.item(), device=stft.device)
        for r in rate]
    # [B]
    steplen = torch.tensor([len(s) for s in steps], device=steps.device, dtype=torch.long)
    # L
    slen = steplen.amax().item()
    # [B, L], replicate padding
    steps = torch.stack([F.pad(s, [slen - len(s)], value=s[-1].item()) for s in steps], dim=0)
    # [B, num_freq, 1]
    phase_0 = stft[..., :1].angle()
    # [B, num_freq, T / hop + 2]
    stft = F.pad(stft, [0, 2])
    # [B, num_freq, L]
    index = steps[:, None].repeat(num_freq)
    stft_0 = stft.gather(-1, index.long())
    stft_1 = stft.gather(-1, (index + 1).long())
    # compute angle
    angle_0 = stft_0.angle()
    angle_1 = stft_1.angle()
    # compute norm
    norm_0 = stft_0.abs()
    norm_1 = stft_1.abs()
    # [B, num_freq, L]
    phase = angle_1 - angle_0 - phase_adv[None, :, None]
    phase = phase - 2 * np.pi * torch.round(phase / (2 * np.pi))
    # phase accumulation
    phase = phase + phase_adv[None, :, None]
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)
    # [num_freq]
    alphas = steps % 1.0
    # [B, num_freq, T / hop / rate], linear interpolation
    mag = alphas * norm_1 + (1 - alphas) * norm_0
    # to complex tensor
    stretched = torch.polar(mag, phase_acc)
    return stretched, steplen


class PitchShift(nn.Module):
    """Pitch-shift batch support.
    """
    def __init__(self,
                 sr: int,
                 step_min: int,
                 step_max: int,
                 bins_per_octave: int = 12,
                 fft: int = 512,
                 hop: int = 128,
                 win: int = 512,
                 window: Optional[torch.Tensor] = None):
        """Initializer.
        Args:
            sr: sampling rate.
            step_min, step_max: minimum, maximum steps.
            bins_per_octave: the number of the midi bins in an octave.
        """
        super().__init__()
        self.sr = sr
        # [step_max - step_min + 1]
        steps = torch.arange(step_min, step_max + 1)
        # speed rate
        rate = 2 ** (-steps.float() / bins_per_octave)
        # resampling kernels
        self.resamples = nn.ModuleDict({
            str(s.item()): torchaudio.transforms.Resample(orig_freq.item(), sr)
            for s, orig_freq in zip(tqdm(steps.long(), desc='resampler'), (sr / rate).long())})
        # alias
        self.fft, self.hop, self.win = fft, hop, win
        if window is None:
            window = torch.hann_window(win)
        self.register_buffer('window', window, persistent=False)

    def forward(self, wav: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        """Shift the pitch.
        Args:
            wav: [torch.float32; [B, T]], speech signal, [-1, 1]-ranged.
            steps: [torch.long; [B]], shift steps.
        Returns:
            [torch.float32; [B, T]], shifted, [-1, 1]-ranged.
        """
        # T
        _, timesteps = wav.shape
        # [B, fft // 2 + 1, T / hop]
        stft = torch.stft(wav, self.fft, self.hop, self.win, self.window, return_complex=True)
        # [B]
        rate = 2 ** (-steps.float() / self.bins_per_octave)
        # [B, fft // 2 + 1, T / hop / rate], [B]
        stretched, steplen = phase_vocoder(stft, rate)
        # [B, T / rate]
        shifted = torch.istft(stretched, self.fft, self.hop, self.win, self.window)
        # B x[T]
        resampled = [
            self.resamples[str(step.item())](shift[:slen.item()])
            for step, slen, shift in zip(steps, steplen, shifted)]
        return torch.stack([
            F.pad(r[:timesteps], [0, max(0, timesteps - len(r))])
            for r in resampled], dim=0)

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def phase_vocoder(stft: torch.Tensor, rate: torch.Tensor, hop: int):
    """Speed up in time without modifying pitch, ref:torchaudio.functional.phase_vocoder.
    Args:
        stft: [torch.complex64; [B, fft // 2 + 1, T]], fourier features.
        rate: [torch.float32; [B]], speed rate.
    """
    device = stft.device
    # fft // 2 + 1
    _, num_freq, timesteps  = stft.shape
    # [num_freq]
    phase_adv = torch.linspace(0, np.pi * hop, num_freq, device=device)
    # B x [L(=T / rate)]
    steps = [
        torch.arange(0, timesteps, r.item(), device=device)
        for r in rate]
    # [B]
    steplen = torch.tensor([len(s) for s in steps], device=device, dtype=torch.long)
    # L
    slen = steplen.amax().item()
    # [B, L], replicate padding
    steps = torch.stack([
        F.pad(s, [0, slen - len(s)], value=s[-1].item()) for s in steps], dim=0)
    # [B, num_freq, 1]
    phase_0 = stft[..., :1].angle()
    # [B, num_freq, T + 2]
    stft = F.pad(stft, [0, 2])
    # [B, num_freq, L]
    index = steps[:, None].repeat(1, num_freq, 1)
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
    # [B, 1, L]
    alphas = steps[:, None] % 1.0
    # [B, num_freq, L], linear interpolation
    mag = alphas * norm_1 + (1 - alphas) * norm_0
    # to complex tensor
    stretched = torch.polar(mag, phase_acc)
    return stretched


class PitchShift(nn.Module):
    """Pitch-shift batch support.
    """
    def __init__(self,
                 sr: int,
                 bins_per_octave: int = 12,
                 fft: int = 512,
                 hop: int = 128,
                 win: int = 512,
                 window: Optional[torch.Tensor] = None):
        """Initializer.
        Args:
            sr: sampling rate.
            bins_per_octave: the number of the midi bins in an octave.
        """
        super().__init__()
        # alias
        self.sr, self.bins_per_octave = sr, bins_per_octave
        self.fft, self.hop, self.win = fft, hop, win
        if window is None:
            window = torch.hann_window(win)
        self.register_buffer('window', window, persistent=False)

    @torch.no_grad()
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
        stretched = phase_vocoder(stft, rate, self.hop)
        # [B, T / rate]
        shifted = torch.istft(stretched, self.fft, self.hop, self.win, self.window)
        # TODO: antialias
        # [B, 1, T]
        o = torch.cat([
            F.interpolate(shifts[None, None, :s.item()], timesteps, mode='linear')
            for shifts, s in zip(shifted, (timesteps / rate).long())], dim=0)
        return o.squeeze(dim=1).clamp(-1., 1.)

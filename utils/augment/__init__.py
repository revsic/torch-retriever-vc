from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .peq import ParametricEqualizer
from .pshift import PitchShift

from config import Config


class Augment(nn.Module):
    """Waveform augmentation.
    """
    def __init__(self, config: Config):
        """Initializer.
        Args:
            config: Nansy configurations.
        """
        super().__init__()
        self.config = config
        self.peq = ParametricEqualizer(
            config.model.sr, config.model.win)
        self.register_buffer(
            'window',
            torch.hann_window(config.model.win),
            persistent=False)
        f_min, f_max, peaks = \
            config.train.cutoff_lowpass, \
            config.train.cutoff_highpass, config.train.num_peak
        # peaks except frequency min and max
        self.register_buffer(
            'peak_centers',
            f_min * (f_max / f_min) ** (torch.arange(peaks + 2)[1:-1] / (peaks + 1)),
            persistent=False)
        # batch shifting supports
        self.pitch_shift = PitchShift(config.model.sr, config.train.bins_per_octave)

    def forward(self,
                wavs: torch.Tensor,
                pitch_shift: Optional[torch.Tensor] = None,
                quality_power: Optional[torch.Tensor] = None,
                gain: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Augment the audio signal, random pitch, formant shift and PEQ.
        Args:
            wavs: [torch.float32; [B, T]], audio signal.
            pitch_shift: [torch.float32; [B]], pitch shifts.
            quality_power: [torch.float32; [B, num_peak + 2]],
                exponents of quality factor, for PEQ.
            gain: [torch.float32; [B, num_peak + 2]], gain in decibel.
        Returns:
            [torch.float32; [B, T]], augmented.
        """
        # B
        bsize, _ = wavs.shape
        # [B, F, T / S], complex64
        fft = torch.stft(
            wavs,
            self.config.model.fft,
            self.config.model.hop,
            self.config.model.win,
            self.window,
            return_complex=True)
        # PEQ
        if quality_power is not None:
            # alias
            q_min, q_max = self.config.train.q_min, self.config.train.q_max
            # [B, num_peak + 2]
            q = q_min * (q_max / q_min) ** quality_power
            if gain is None:
                # [B, num_peak]
                gain = torch.zeros_like(q[:, :-2])
            # [B, num_peak]
            center = self.peak_centers[None].repeat(bsize, 1)
            # [B, F]
            peaks = torch.prod(
                self.peq.peaking_equalizer(center, gain[:, :-2], q[:, :-2]), dim=1)
            # [B, F]
            lowpass = self.peq.low_shelving(
                self.config.train.cutoff_lowpass, gain[:, -2], q[:, -2])
            highpass = self.peq.high_shelving(
                self.config.train.cutoff_highpass, gain[:, -1], q[:, -1])
            # [B, F]
            filters = peaks * highpass * lowpass
            # [B, F, T / S]
            fft = fft * filters[..., None]
        # [B, T]
        out = torch.istft(
            fft,
            self.config.model.fft,
            self.config.model.hop,
            self.config.model.win,
            self.window).clamp(-1., 1.)
        # max value normalization
        out = out / out.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-7)
        if pitch_shift is not None:
            # [B], in midi-range
            steps = (12 * pitch_shift.log2()).long()
            # [B, T]
            out = self.pitch_shift.forward(out, steps)
        return out

    def sample_like(self, signal: torch.Tensor) -> List[torch.Tensor]:
        """Sample augmentation parameters.
        Args:
            signal: [torch.float32; [B, T]], speech signal.
        Returns:
            augmentation parameters.
        """
        # [B]
        bsize, _ = signal.shape
        def sampler(ratio):
            shifts = torch.rand(bsize, device=signal.device) * (ratio - 1.) + 1.
            # flip
            flip = torch.rand(bsize) < 0.5
            shifts[flip] = shifts[flip] ** -1
            return shifts
        # sample shifts
        ps = sampler(self.config.train.pitch_shift)
        # parametric equalizer
        peaks = self.config.train.num_peak
        # quality factor
        power = torch.rand(bsize, peaks + 2, device=signal.device)
        # gains
        g_min, g_max = self.config.train.g_min, self.config.train.g_max
        gain = torch.rand(bsize, peaks + 2, device=signal.device) * (g_max - g_min) + g_min
        return ps, power, gain

    @torch.no_grad()
    def augment(self, signal: torch.Tensor) -> torch.Tensor:
        """Augment the speech.
        Args:
            signal: [torch.float32; [B, T]], segmented speech.
        Returns:
            [torch.float32; [B, T]], speech signal.
        """
        # B
        bsize, _ = signal.shape
        saves = None
        while saves is None or len(saves) < bsize:
            # [B] x 4
            pshift, power, gain = self.sample_like(signal)
            # [B, T]
            out = self.forward(signal, pshift, power, gain)
            # for covering unexpected NaN
            nan = out.isnan().any(dim=-1)
            if not nan.all():
                # save the outputs for not-nan inputs
                if saves is None:
                    saves = out[~nan]
                else:
                    saves = torch.cat([saves, out[~nan]], dim=0)
        # [B, T]
        return saves[:bsize]

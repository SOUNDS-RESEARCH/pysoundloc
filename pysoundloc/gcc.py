"""
This module contains functions for performing cross-correlation
between signals using the GCC-PHAT method
"""

import numpy as np
import torch

from math import ceil, log2
from scipy.signal import correlate, correlation_lags

from .stft import CrossSpectra

EPS = 1e-7


def temporal_cross_correlation(x1, x2, sr):
    
    # Normalize signals for a normalized correlation
    # https://github.com/numpy/numpy/issues/2310
    x1 = (x1 - np.mean(x1)) / (np.std(x1) * len(x1))
    x2 = (x2 - np.mean(x2)) / (np.std(x2) * len(x2))
    
    cc = correlate(x1, x2, mode="same")
    lag_indexes = correlation_lags(x1.shape[0], x2.shape[0], mode="same")

    cc = np.abs(cc)
    
    return cc, lag_indexes/sr


# TODO: Deprecate this version
def gcc_phat_batch(x1, x2, sr, abs=False):
    if isinstance(x1, np.ndarray):
        return _gcc_phat_numpy_batch(x1, x2, sr, abs=abs)
    elif isinstance(x1, torch.Tensor):
        return _gcc_phat_torch_batch(x1, x2, sr, abs=abs)


def _gcc_phat_torch_batch(x1, x2, sr, use_pow2_fft=False, abs=True,
                          phase_transform=True):
    """
    This function computes the ofsret between the signal sig and the reference signal x2
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT) method.
    Implementation based on http://www.xavieranguera.com/phdthesis/node92.html
    """
    
    n = x1.shape[1] # + x2.shape[0]
    if use_pow2_fft:
        n = 2**(ceil(log2(n))) # Get next power of 2 (faster fft)

    X1 = torch.fft.rfft(x1, n=n, dim=1)
    X2 = torch.fft.rfft(x2, n=n, dim=1)
    R = X1 * torch.conj(X2)
    Gphat = R / (EPS + torch.abs(R))
    cc = torch.fft.irfft(Gphat, n=n, dim=1)
    max_delay = n // 2

    cc = torch.cat((cc[:, -max_delay:], cc[:, :max_delay+1]), dim=1)
    if abs:
        cc = torch.abs(cc)

    indxs = torch.zeros(cc.shape[1])
    indxs[0:max_delay] = - torch.arange(max_delay, 0, -1)
    indxs[max_delay:] = torch.arange(0, max_delay + 1)
    indxs = indxs/sr

    return cc, indxs


def _gcc_phat_numpy_batch(x1, x2, sr, use_pow2_fft=False, abs=True):
    """
    Batch version of _gcc_phat_numpy
    """
    
    n = x1.shape[1] # + x2.shape[0]
    if use_pow2_fft:
        n = 2**(ceil(log2(n))) # Get next power of 2 (faster fft)

    X1 = np.fft.rfft(x1, n=n, axis=1)
    X2 = np.fft.rfft(x2, n=n, axis=1)
    R = X1 * np.conj(X2)
    Gphat = R / (EPS + np.abs(R))
    cc = np.fft.irfft(Gphat, n=n, axis=1)
    max_delay = n // 2

    cc = np.concatenate((cc[:, -max_delay:], cc[:, :max_delay+1]), axis=1)
    if abs:
        cc = np.abs(cc)

    indxs = np.zeros(cc.shape[1])
    indxs[0:max_delay] = - np.arange(max_delay, 0, -1)
    indxs[max_delay:] = np.arange(0, max_delay + 1)
    indxs = indxs/sr

    return cc, indxs


class Gcc(CrossSpectra):
    def __init__(self, sr, abs=True, mode="stft", n_dft=4096, hop_size=2048,
                 window_length=None, window="hann", phase_transform=True):
        super().__init__(n_dft, hop_size, window_length, remove_dc=False,
                         phase_transform=phase_transform, window=window)
        self.sr = sr
        self.abs = abs
        self.mode = mode

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        if self.mode == "stft":
            return self._gcc_stft(x1, x2)
        elif self.mode == "dft":
            return self._gcc_dft(x1, x2)
        
    def _gcc_dft(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Non-framed mode of GCC, which processes the whole signals simultaneously
        """
        return _gcc_phat_torch_batch(x1, x2, self.sr, abs=self.abs,
                                     phase_transform=self.phase_transform)

    def _gcc_stft(self, x1: torch.Tensor, x2: torch.Tensor):
        """Split the signals x1 and x2 into frames of size n_dft and hop size hop_size,
        and compute the GCC-PHAT for them. Then, average across all frames. 

        Args:
            x1 (torch.Tensor): first signal
            x2 (torch.Tensor): second signal

        Returns:
            (torch.Tensor, torch.Tensor): (GCC-PHAT of shape (batch_size, n_dft), temporal indices)
        """
        n = self.window_length

        x = torch.stack([x1, x2], dim=1)

        cc_freq = super().forward(x)[:, 0, 1] # Remove autocorrelation and repeated correlation
        
        cc_freq = cc_freq.mean(dim=1) # Average across frames

        cc_time = torch.fft.irfft(cc_freq, n=n)
        
        max_delay = n // 2
        cc_time = torch.cat((cc_time[:, -max_delay:], cc_time[:, :max_delay]), dim=1)

        if self.window == "hann":
            window = torch.hann_window(self.window_length, device=x.device).unsqueeze(0)
        else:
            raise ValueError("Only hann window currently supported")
        # cc_time *= window
        
        if self.abs:
            cc_time = cc_time.abs()
        
        cc_delays = get_correlation_delays(n, self.sr)
        
        return cc_time, cc_delays


def get_correlation_delays(n, sr):
    delays = torch.zeros(n)
    max_delay = n // 2
    delays[0:max_delay] = - torch.arange(max_delay, 0, -1)
    delays[max_delay:] = torch.arange(0, max_delay)
    delays = delays/sr
    return delays

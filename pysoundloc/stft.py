import torch

from torch.nn import Module

EPS = 1e-7


class StftArray(Module):
    def __init__(self, n_dft=1024, hop_size=512, window_length=None,
                 onesided=True, is_complex=True, complex_as_channels=False,
                 mag_only=False, phase_only=False, real_only=False,
                 window="hann", remove_dc=True):

        super().__init__()

        self.n_dft = n_dft
        self.hop_size = hop_size
        self.onesided = onesided
        self.is_complex = is_complex
        self.complex_as_channels = complex_as_channels

        self.mag_only = mag_only
        self.phase_only = phase_only
        self.real_only = real_only
        self.window_length = n_dft if window_length is None else window_length
        self.window = window
        self.remove_dc = remove_dc

        if window not in ["hann", None]:
            raise ValueError("Only 'hann' and None windows are currently supported")

    def forward(self, x: torch.Tensor):
        "Expected input has shape (batch_size, n_channels, time_steps)"

        input_shape = x.shape

        if len(input_shape) == 3:
            # (batch_size, n_channels, time_steps) => Microphone array
            # Collapse channels into batch
            x = x.flatten(end_dim=1)

        
        window = self._create_window(x.device)

        y = torch.stft(x, self.n_dft, hop_length=self.hop_size, 
                       onesided=self.onesided, return_complex=True,
                       win_length=self.window_length, window=window)
        
        if self.remove_dc:
            y = y[:, 1:] # Remove DC component (f=0hz)

        y = y.transpose(1, 2)
        # y.shape == (batch_size*channels, time, freqs)

        if len(input_shape) == 3:
            batch_size, num_channels, _ = input_shape
            # De-collapse first dim into batch and channels
            y = y.unflatten(0, (batch_size, num_channels))

        if self.mag_only:
            return y.abs()
        if self.phase_only:
            return y.angle()
        if self.real_only:
            return y.real

        if not self.is_complex:
            y = _complex_to_real(y, self.complex_as_channels)

        return y

    def _create_window(self, device):
        if self.window == "hann":
            return torch.hann_window(self.window_length, device=device)
        elif self.window is None:
            return None


class CrossSpectra(StftArray):
    def __init__(self, n_dft=1024, hop_size=512, window_length=None, window="hann",
                 onesided=True, is_complex=True, complex_as_channels=False,
                 phase_only=False, avg_dim=None, remove_dc=True, phase_transform=False):        
        super().__init__(n_dft, hop_size, window_length=window_length,
                         onesided=onesided, remove_dc=remove_dc, window=window)

        self._is_complex = is_complex # _ is added not to conflict with StftArray
        self.complex_as_channels = complex_as_channels
        self.phase_only = phase_only
        self.avg_dim = avg_dim
        self.phase_transform = phase_transform


    def forward(self, X) -> torch.Tensor:
        "Expected input has shape (batch_size, n_channels, time_steps)"
        batch_size, n_channels, time_steps = X.shape

        stfts = super().forward(X)
        # (batch_size, n_channels, n_time_bins, n_freq_bins)
        y = []

        # Compute the cross-spectrum between each pair of channels
        
        stfts_col = stfts.unsqueeze(2)
        stfts_row = stfts.unsqueeze(1)

        y = stfts_col*stfts_row.conj()

        if self.phase_transform:
            y /= stfts_col.abs()*stfts_row.conj().abs() + EPS

        if self.phase_only:
            return y.angle()

        if not self._is_complex:
            y = _complex_to_real(y, self.complex_as_channels)

        if self.avg_dim is not None:
            y = y.mean(dim=self.avg_dim)
        return y


def _complex_to_real(x, as_channels=False): 
    y = torch.view_as_real(x)
    if as_channels:
        # Merge channels and real and imaginary parts (last dim) as separate input channels
        y = y.transpose(2, -1).flatten(start_dim=1, end_dim=2)

    return y

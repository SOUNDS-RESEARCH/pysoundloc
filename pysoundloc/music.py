import numpy as np
import torch

from .tdoa import compute_theoretical_tdoa_grids
from .energy import compute_theoretical_energy_ratio_grids
from .stft import CrossSpectra


def music(mic_signals, mic_coords, room_dims, sr, n_sources, n_grid_points=25,
          n_ref_mic=0, normalized=True, n_dft=1024, hop_size=512):
    """MUltiple SIgnal Classification method for positional sound source localization

    Args:
        mic_signals: Numpy array or torch tensor
        mic_coords: Numpy array or torch tensor
        room_dims: Numpy array or torch tensor
        sr (float): Sampling rate
        n_grid_points: Grid resolution 
        n_ref_mic (int, optional): Reference microphone to compute steering vectors from. Defaults to 0.
        normalized (bool, optional): Whether to normalize per frequency.
            See https://ieeexplore.ieee.org/document/6774907 for details. Defaults to True.

    Returns:
        dict: Dictionary containing "source_coordinates" and "grid" keys
    """

    batch_size, n_mics = mic_signals.shape[:2]
    cross_spectra_func = CrossSpectra(n_dft=n_dft, hop_size=hop_size)
    
    # If numpy arrays are provided, convert them to torch tensors
    to_numpy = False
    if isinstance(mic_signals, np.ndarray):
        mic_signals = torch.from_numpy(mic_signals)
        mic_coords = torch.from_numpy(mic_coords)
        room_dims = torch.from_numpy(room_dims)
        to_numpy = True

    steering_vectors = compute_theoretical_steering_vector_grids(mic_coords, room_dims, sr,
                                                                 n_grid_points, n_ref_mic,
                                                                 n_dft_bins=n_dft)

    cross_spectra = cross_spectra_func(mic_signals)
    # cross_spectra.shape = (batch, n_mics, n_mics, time, freq)

    cross_spectra = cross_spectra.mean(dim=3) # Average across time
    
    cross_spectra = cross_spectra.movedim(-1, 1) # Move freq axis to front, as torch.linalg.eigh
                                                 # expects the matrix dims to be the last two ones
    
    eigenvalues, eigenvectors = torch.linalg.eigh(cross_spectra)

    eigenvectors = eigenvectors.movedim(1, -1)

    n_noise_eigenvectors = n_mics - n_sources

    spatial_spectrum = torch.zeros(batch_size, n_grid_points, n_grid_points)

    for i in range(n_noise_eigenvectors):
        # Compute the dot product between the noise eigenvector and
        # the candidate steering vector
        eigenvector = eigenvectors[:, i].unsqueeze(2).unsqueeze(2)
        eigenvector = eigenvector.expand(steering_vectors.shape)
        dot_result = (eigenvector*steering_vectors.conj()).sum(dim=1).abs()
        spatial_spectrum += dot_result.mean(dim=-1)
        # Average across all frequencies
        # TODO: Normalize properly
    # spatial_spectrum = spatial_spectrum.mean(dim=-1)

    return spatial_spectrum


def compute_theoretical_steering_vector_grids(mic_coords, room_dims, sr,
                                              n_grid_points_per_axis=25,
                                              n_reference_mic=0,
                                              n_dft_bins=512):
    
    batch_size, n_mics = mic_coords.shape[:2]
    
    freq_grid = torch.ones(
        batch_size,
        n_grid_points_per_axis,
        n_grid_points_per_axis,
        n_dft_bins//2)*torch.arange(n_dft_bins//2)

    steering_grids = []
    for i in range(n_mics):
        tdoa_grids = compute_theoretical_tdoa_grids(
            mic_coords[:, n_reference_mic], mic_coords[:, i],
            room_dims, n_grid_points_per_axis
        )

        energy_grids = compute_theoretical_energy_ratio_grids(
            mic_coords[:, n_reference_mic], mic_coords[:, i],
            room_dims, n_grid_points_per_axis
        )
        
        phase = torch.exp(tdoa_grids.unsqueeze(-1)*(-2j*torch.pi)*(sr*freq_grid/n_dft_bins))
        steering_grids.append(phase) # energy_grids.unsqueeze(-1)*phase

    if isinstance(steering_grids[0], torch.Tensor):
        return torch.stack(steering_grids, dim=1)
    else:
        return np.stack(steering_grids, axis=1)

import numpy as np
import torch

from .utils.grid import create_fixed_size_grids

from .gcc import gcc_phat_batch
from .utils.math import SPEED_OF_SOUND, compute_distance_grids


def compute_theoretical_tdoa_grids(mic_0_coords, mic_1_coords,
                                   room_dims, n_grid_points_per_axis=25, n_dims=None):
    """Create a grid where each point represents a position in a room with room_dims
    dimensions.
    """

    if n_dims is not None:
        mic_0_coords = mic_0_coords[..., :2]
        mic_1_coords = mic_1_coords[..., :2]
        room_dims = room_dims[..., :2]

    dist_0 = compute_distance_grids(mic_0_coords, room_dims,
                                    n_grid_points_per_axis)
    dist_1 = compute_distance_grids(mic_1_coords, room_dims,
                                    n_grid_points_per_axis)

    return (dist_0 - dist_1)/SPEED_OF_SOUND


def compute_tdoa_grids(mic_0_signals, mic_1_signals, sr,
                       mic_0_coords, mic_1_coords,
                       room_dims, n_grid_points=25):
    if isinstance(mic_0_signals, torch.Tensor):
        lib = torch
        argmax = lambda x: torch.argmax(x, dim=1)
    elif isinstance(mic_0_signals, np.ndarray):
        lib = np
        argmax = lambda x: np.argmax(x, axis=1)
    else:
        raise ValueError("Signals must be either torch tensors or numpy arrays")
    
    batch_size = room_dims.shape[0]

    grids = create_fixed_size_grids(room_dims, n_grid_points)[0]
    theoretical_tdoa_grids = compute_theoretical_tdoa_grids(mic_0_coords, mic_1_coords, grids)

    cc_vectors, indxs = gcc_phat_batch(mic_0_signals, mic_1_signals, sr)
    tdoa_indxs = argmax(lib.abs(cc_vectors))
    tdoas = lib.stack([indxs[tdoa_indx] for tdoa_indx in tdoa_indxs])

    ls_error_grids = (theoretical_tdoa_grids - tdoas.reshape((batch_size, 1, 1)))**2
    
    return ls_error_grids

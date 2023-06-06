import torch
import numpy as np

from .utils.math import grid_argmax, grid_argmin
from .music import music
from .srp import compute_pairwise_srp_grids
from .tdoa import compute_tdoa_grids


def srp_phat(mic_signals, mic_coords, room_dims, sr, n_grid_points,
             mode="max_correlation_neighbours",
             return_per_mic=False, return_grid_pairs=False, n_correlation_neighbours=10,
             phase_transform=True, gcc_mode="stft", n_dft=8192):

    estimated_locations, grid_sum = _grid_ssl(compute_pairwise_srp_grids,
        mic_signals, mic_coords, room_dims, sr, n_grid_points,
        return_per_mic=return_per_mic, return_grid_pairs=return_grid_pairs,
        grid_type="likelihood", mode=mode, n_correlation_neighbours=n_correlation_neighbours,
        phase_transform=phase_transform, gcc_mode=gcc_mode, n_dft=n_dft
    )

    return {
        "source_coordinates": estimated_locations,
        "grid": grid_sum
    }


def tdoa_least_squares_ssl(mic_signals, mic_coords, room_dims, sr, n_grid_points,
                           return_per_mic=False, return_grid_pairs=False):

    estimated_locations, grid_sum = _grid_ssl(compute_tdoa_grids,
        mic_signals, mic_coords, room_dims, sr, n_grid_points,
        return_per_mic=return_per_mic, return_grid_pairs=return_grid_pairs,
        grid_type="error"
    )    

    return estimated_locations, grid_sum


def _grid_ssl(grid_func, mic_signals, mic_coords, room_dims, sr, n_grid_points,
              return_per_mic=False, return_grid_pairs=False, grid_type="likelihood", **kwargs):
    "Generic class for generating a heatmap between all microphone pairs"

    if isinstance(mic_signals, torch.Tensor):
        lib = torch
        sum_func = lambda x: torch.sum(x, dim=2)
    elif isinstance(mic_signals, np.ndarray):
        lib = np
        sum_func = lambda x: np.sum(x, axis=2)
    
    batch_size, n_mics, _ = mic_signals.shape 
    
    grid_sum = lib.zeros((batch_size, n_grid_points, n_grid_points))
    if return_grid_pairs:
        grids = lib.zeros((batch_size, n_mics, n_mics, n_grid_points, n_grid_points)) # One grid per microphone pair

    # 1. Compute grids for all microphone pairs
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            grid = grid_func(
                    mic_signals[:, i], mic_signals[:, j],
                    sr, mic_coords[:, i], mic_coords[:, j],
                    room_dims, n_grid_points=n_grid_points,
                    **kwargs
            )
            
            grid_sum += grid
            
            if return_grid_pairs:
                grids[:, i, j] = grid
                grids[:, j, i] = grid
    
    # 2. Estimate source coordinates by picking the minimum point within the sum grid
    if grid_type == "error":
        estimated_locations = grid_argmin(grid_sum, room_dims)
    elif grid_type == "likelihood":
        # Negate the grid, as the grid_argmin function will search for a minimum
        estimated_locations = grid_argmax(grid_sum, room_dims)

    if return_grid_pairs:
        return estimated_locations, grids
    elif return_per_mic:
        return estimated_locations, sum_func(grids)
    else:
        return estimated_locations, grid_sum

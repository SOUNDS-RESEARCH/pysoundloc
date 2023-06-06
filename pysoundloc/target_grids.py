import numpy as np
import torch

from .utils.grid import create_fixed_size_grids
from .utils.math import SPEED_OF_SOUND, point_to_rect_min_distance
from .tdoa import compute_theoretical_tdoa_grids

# This file contains functions used as targets for training models


def create_target_gaussian_grids(
    source_coords, room_dims, n_grid_points_per_axis=25, sigma=1, log=False,
    squared=False):
    """
    Create a heatmap where the source is stronger value is
    located at source_coords
    """
    
    if isinstance(source_coords, np.ndarray):
        lib = np
        max_func = lambda x: np.max(x, axis=0)
    elif isinstance(source_coords, torch.Tensor):
        lib = torch
        max_func = lambda x: torch.max(x, dim=0)[0]
    
    coordinate_grid = create_fixed_size_grids(room_dims, n_grid_points_per_axis)[0]

    n_sources = source_coords.shape[1]
    likelihood_grids = lib.stack([
        _compute_likelihood_grids_for_single_source(coordinate_grid, source_coords[:, i], sigma, log=log,
            squared=squared)
        for i in range(n_sources)
    ])
    
    likelihood_grid = max_func(likelihood_grids)

    return likelihood_grid


def create_target_hyperbolic_grids(source_coords,
                                   mic_0_coords,
                                   mic_1_coords,
                                   room_dims,
                                   n_grid_points_per_axis=25,
                                   sigma=1,
                                   log=False,
                                   squared=False):
    
    if isinstance(source_coords, np.ndarray):
        norm = lambda x: np.linalg.norm(x, axis=1)
        lib = np
    elif isinstance(source_coords, torch.Tensor):
        norm = lambda x: torch.linalg.norm(x, dim=1)
        lib = torch

    # 1. Compute TDOA for each candidata position
    tdoa_grids = compute_theoretical_tdoa_grids(mic_0_coords, mic_1_coords,
                                                room_dims, n_grid_points_per_axis)

    # 2. Compute TDOA for the actual source position
    # TODO: This only works for 2D (planar) localization

    if lib == torch:
        likelihood_grid = torch.zeros_like(tdoa_grids, device=tdoa_grids.device)
    elif lib == np:
        likelihood_grid = np.zeros_like(tdoa_grids)

    n_sources = source_coords.shape[1]
    for i in range(n_sources):
        source_coords_i = source_coords[:, i, :2]
        dist_0 = norm(mic_0_coords - source_coords_i)
        dist_1 = norm(mic_1_coords - source_coords_i)

        tdoa = (dist_0 - dist_1)/SPEED_OF_SOUND

        likelihood_grid_i = lib.abs(tdoa[:, np.newaxis, np.newaxis] - tdoa_grids)
        if squared:
            likelihood_grid_i = likelihood_grid_i**2
        likelihood_grid_i = -likelihood_grid_i/lib.max(likelihood_grid_i) # negate and normalize

        if not log: # Applying the log is equivalent to not applying the exp
            likelihood_grid_i = lib.exp(likelihood_grid_i/(sigma**2))
        likelihood_grid += likelihood_grid_i

    return likelihood_grid


def pick_peak(likelihood_grid, room_dims, n_sources=1):
    """This is a peak-picking function which estimates the coordinates of
    a source by picking the maximum value in likelihood_grid and scaling it
    to the appropriate room dimensions
    """
    # TODO: add multiple sources/unknown sources (n_sources=None)

    x_cell_resolution_in_meters = room_dims[0]/likelihood_grid.shape[0]
    y_cell_resolution_in_meters = room_dims[1]/likelihood_grid.shape[1]

    if n_sources != 1:
        raise NotImplementedError("Only one source is currently implemented")
    
    if isinstance(likelihood_grid, np.ndarray):
        idx = np.unravel_index(likelihood_grid.argmax(), likelihood_grid.shape)
        cell_resolution_in_meters = np.array([
            x_cell_resolution_in_meters, y_cell_resolution_in_meters]
        )
    elif isinstance(likelihood_grid, torch.Tensor):
        idx = (likelihood_grid==torch.max(likelihood_grid)).nonzero()
        cell_resolution_in_meters = torch.Tensor([
            x_cell_resolution_in_meters, y_cell_resolution_in_meters]
        )

    return idx*cell_resolution_in_meters + cell_resolution_in_meters/2


def _compute_likelihood_grids_for_single_source(coordinate_grids,
                                                source_coordinates,
                                                sigma=1, log=False, squared=False):
    if isinstance(coordinate_grids, torch.Tensor):
        lib = torch
    elif isinstance(coordinate_grids, np.ndarray):
        lib = np
    
    source_coordinates_grid = lib.zeros_like(coordinate_grids)
    source_coordinates_grid[:, 0] = source_coordinates[:, 0, np.newaxis, np.newaxis]
    source_coordinates_grid[:, 1] = source_coordinates[:, 1, np.newaxis, np.newaxis]

    # source_coordinates_grid has the centre of the coordinates,
    # but each cell encodes a square.
    # The code below computes the distance between the each cell rectangular region and the source_coordinates,
    
    # dx = coordinate_grids[0, 0, 0]
    # dy = coordinate_grids[1, 0, 0]
    # distance_grid = point_to_rect_min_distance(source_coordinates[0], source_coordinates[1],
    #                                            coordinate_grids[0] - dx,
    #                                            coordinate_grids[0] + dx,
    #                                            coordinate_grids[1] - dy,
    #                                            coordinate_grids[1] + dy)

    distance_grid = lib.sum((coordinate_grids - source_coordinates_grid)**2, axis=1)
    if not squared:
        distance_grid = lib.sqrt(distance_grid)

    likelihood_grid = -distance_grid/(sigma**2)
    if not log: # Applying the log is equivalent to not applying the exp
        likelihood_grid = lib.exp(likelihood_grid)
    
    return likelihood_grid

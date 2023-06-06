import numpy as np
import torch

from torch import Tensor


def create_fixed_resolution_grid(room_dims, grid_resolution_in_meters, mode="numpy"):
    if mode == "numpy":
        lib = np
        tile_func = np.tile
    elif mode == "torch":
        lib = torch
        tile_func = lambda x, n: torch.tile(x, (n,))
    
    x_coordinates = np.arange(0, room_dims[0], grid_resolution_in_meters)
    y_coordinates = np.arange(0, room_dims[1], grid_resolution_in_meters)

    n_grid_rows = x_coordinates.shape[0]
    n_grid_cols = y_coordinates.shape[0]

    x_coordinate_grid = tile_func(x_coordinates.reshape((n_grid_rows, 1)),
                                  n_grid_cols)
    y_coordinate_grid = tile_func(y_coordinates.reshape((n_grid_cols, 1)),
                                  n_grid_rows).T
    
    grid = lib.stack([x_coordinate_grid, y_coordinate_grid])
    
    return grid, (x_coordinates, y_coordinates)


def create_fixed_size_grids(room_dims, n_grid_points_per_axis):
    if type(room_dims) == list:
        room_dims = np.array(room_dims)

    if len(room_dims.shape) == 1:
        room_dims = room_dims.reshape((1, room_dims.shape[0])) # Batchify
    
    if isinstance(room_dims, torch.Tensor):
        return _create_fixed_size_grids_torch(room_dims, n_grid_points_per_axis)
    elif isinstance(room_dims, np.ndarray):
        return _create_fixed_size_grids_numpy(room_dims, n_grid_points_per_axis)
    else:
        raise ValueError("room_dims must be numpy array or torch tensor")


def _create_fixed_size_grids_numpy(room_dims, n_grid_points_per_axis):
    """
    Batch version for 'create_fixed_size_grid' function
    """
    
    batch_size = room_dims.shape[0]
    
    # 1. Get cell resolution
    x_cell_resolution_in_meters = room_dims[:, 0]/n_grid_points_per_axis
    y_cell_resolution_in_meters = room_dims[:, 1]/n_grid_points_per_axis
    
    # 2. Create coordinate vectors
    def create_ticks(cell_resolution, room_dims):
        ticks = np.linspace(cell_resolution, room_dims,
                            n_grid_points_per_axis) - cell_resolution/2
        return ticks.T
    x_coordinates = create_ticks(x_cell_resolution_in_meters, room_dims[:, 0])
    y_coordinates = create_ticks(y_cell_resolution_in_meters, room_dims[:, 1])

    x_coordinates = x_coordinates.reshape((batch_size, n_grid_points_per_axis, 1))
    y_coordinates = y_coordinates.reshape((batch_size, n_grid_points_per_axis, 1))

    # 3. Replicate coordinates and stack them
    x_coordinate_grid = np.tile(x_coordinates, n_grid_points_per_axis)
    y_coordinate_grid = np.tile(y_coordinates, n_grid_points_per_axis).transpose((0, 2, 1))
    
    grid = np.stack([x_coordinate_grid, y_coordinate_grid], axis=1)
    
    return grid, (x_coordinates, y_coordinates)


def _create_fixed_size_grids_torch(room_dims, n_grid_points_per_axis):
    batch_size = room_dims.shape[0]
    
    x_cell_resolution_in_meters = room_dims[:, 0]/n_grid_points_per_axis
    y_cell_resolution_in_meters = room_dims[:, 1]/n_grid_points_per_axis
    
    def create_ticks(cell_resolution, room_dims):
        ticks = linspace(x_cell_resolution_in_meters, room_dims,
                         n_grid_points_per_axis) - cell_resolution/2
        return ticks.T
    
    x_coordinates = create_ticks(x_cell_resolution_in_meters, room_dims[:, 0])
    y_coordinates = create_ticks(y_cell_resolution_in_meters, room_dims[:, 1])
    
    x_coordinates = x_coordinates.reshape((batch_size, n_grid_points_per_axis, 1))
    y_coordinates = y_coordinates.reshape((batch_size, n_grid_points_per_axis, 1))
        
    x_coordinate_grid = torch.tile(x_coordinates,
                                   (n_grid_points_per_axis,))
    y_coordinate_grid = torch.tile(y_coordinates.reshape((batch_size, n_grid_points_per_axis, 1)),
                                   (n_grid_points_per_axis,)).transpose(2, 1)
    
    grid = torch.stack([x_coordinate_grid, y_coordinate_grid], dim=1)
    
    return grid, (x_coordinates, y_coordinates)
    
    
@torch.jit.script
def linspace(start: Tensor, stop: Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.

    Credits to martin-marek: https://github.com/pytorch/pytorch/issues/61292
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

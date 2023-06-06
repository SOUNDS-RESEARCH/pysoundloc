import os
import matplotlib.pyplot as plt
import numpy as np

from ..pysoundloc.target_grid import create_target_likelihood_grid
from ..pysoundloc.visualization import plot_grid


def test_likelihood_map():
    room_dims = [5, 3]
    coordinates_list = np.array([[1, 2], [4, 1.5]])
    n_grid_points_per_axis = 25
    sigma = 0.5
    eps = 1e-7

    likelihood_grid = create_target_likelihood_grid(coordinates_list, room_dims, n_grid_points_per_axis, sigma)
    plot_grid(likelihood_grid, room_dims, source_coords=coordinates_list, log=False)

    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = f"tests/temp/likelihood_grid.png"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    
    plt.savefig(temp_file_path)

    x_cell_resolution_in_meters = room_dims[0]/n_grid_points_per_axis
    y_cell_resolution_in_meters = room_dims[1]/n_grid_points_per_axis
    
    cell_resolution_in_meters = np.array([
        x_cell_resolution_in_meters, y_cell_resolution_in_meters
    ])


    # # Assert source coordinates are the maximum of the heatmap
    # idx_list = (coordinates_list*cell_resolution_in_meters).astype(int)
    # assert (likelihood_grid[idx_list[0][0], idx_list[0][1]] + eps >= likelihood_grid).all()
    # assert (likelihood_grid[idx_list[1][0], idx_list[1][1]] + eps >= likelihood_grid).all()

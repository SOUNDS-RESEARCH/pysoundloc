import os
import matplotlib.pyplot as plt
import numpy as np

from pyroomasync.room import ConnectedShoeBox
from pyroomasync.simulator import simulate

from ..pysoundloc.visualization import plot_grid
from ..pysoundloc.tdoa import compute_pairwise_srp_grids, compute_spatial_correlation_grid



def test_compute_pairwise_srp_grids():
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = f"tests/temp/spatial_correlation_grids.png"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    input_signal = np.random.randn(16000)
    sr = 16000
    rt60 = 1.800
    snr = 30

    room_dims = [6, 6]
    source_coords = np.array([[2, 2]])
    mic_coords = np.array([
        [0.5, 3],
        [5.5, 3],
        [3, 0.5],
        [3, 5.5]
    ])

    room = ConnectedShoeBox(room_dims, fs=sr, rt60=rt60)
    room.add_source(source_coords[0], input_signal)
    room.add_microphone_array(mic_coords)

    mic_signals = simulate(room, snr=snr)[:, 1000:2600]


    mic_0_batch = np.stack(3*[mic_signals[0]])
    mic_0_coords_batch = np.stack(3*[mic_coords[0]])
    mic_1_batch = mic_signals[1:]
    mic_1_coords_batch = mic_coords[1:]

    room_dims_batch = np.array(3*[room_dims])

    grids = compute_pairwise_srp_grids(mic_0_batch, mic_1_batch, sr,
                                              mic_0_coords_batch, mic_1_coords_batch,
                                              room_dims_batch, n_grid_points=1000)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    print(grids.shape)
    for i in range(grids.shape[0]):
        plot_grid(grids[i], room_dims_batch[i], ax=axs[i], log=False,
                 microphone_coords=mic_coords[[0, i+1]], source_coords=source_coords)
    
    plt.savefig(temp_file_path)


def test_compute_spatial_correlation_grid():
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = f"tests/temp/spatial_correlation_grid.png"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    input_signal = np.random.randn(16000)
    sr = 16000
    rt60 = 1.800
    snr = 30

    room_dims = [6, 6]
    source_coords = np.array([[2, 2]])
    mic_coords = np.array([
        [0.5, 3],
        [5.5, 3],
    ])

    room = ConnectedShoeBox(room_dims, fs=sr, rt60=rt60)
    room.add_source(source_coords[0], input_signal)
    room.add_microphone_array(mic_coords)

    mic_signals = simulate(room, snr=snr)[:, 1000:2600]

    room_dims_batch = np.array(3*[room_dims])

    grid = compute_spatial_correlation_grid(mic_signals[0], mic_signals[1], sr,
                                             mic_coords[0], mic_coords[1],
                                             room_dims, n_grid_points=1000)

    print(grid.shape)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    
    plot_grid(grid, room_dims, ax=ax, log=False)
    
    plt.savefig(temp_file_path)

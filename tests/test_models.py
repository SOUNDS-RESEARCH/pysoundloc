import matplotlib.pyplot as plt
import numpy as np
import os
 
from scipy.io import wavfile

from ..pysoundloc.models import music, srp_phat
from ..pysoundloc.visualization import plot_grid

from pyroomasync import ConnectedShoeBox, simulate


def test_music():
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = "tests/temp/music.png"
    temp_wav_path = "tests/temp/sim_music.wav"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    sr = 16000
    input_signal = np.random.randn(48000)
    # t = np.arange(16000)
    # input_signal = np.sin(t*2*np.pi*1000/sr)
    
    
    sr, input_signal = wavfile.read("tests/fixtures/p225_001.wav")
    rt60 = None #0.3
    snr = 10
    n_grid_points = 25

    room_dims = np.array([6, 6, 3])
    source_coords = np.array([[3, 3, 1]])
    mic_coords = np.array([
        [0.5, 1, 1],
        [0.5, 2, 1],
        [0.5, 3, 1],
        [0.5, 4, 1]
    ])

    room = ConnectedShoeBox(room_dims, fs=sr, rt60=rt60)
    room.add_source(source_coords[0], input_signal)
    room.add_microphone_array(mic_coords)

    mic_signals = simulate(room, snr=snr)

    wavfile.write(temp_wav_path, sr, mic_signals.T)

    room_dims_batch = np.array(3*[room_dims])

    grids = music(mic_signals[np.newaxis],
                       mic_coords[np.newaxis, :, :2],
                       room_dims[np.newaxis, :2],
                       sr, len(source_coords), n_grid_points,
                       n_dft=1024, hop_size=512)

    # fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    # print(grids.shape)
    for i in range(grids.shape[0]):
        plot_grid(grids[i], room_dims_batch[i], log=True,
                 microphone_coords=mic_coords, source_coords=source_coords,)
    
    plt.savefig(temp_file_path)


def test_srp():
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = "tests/temp/srp.png"
    temp_wav_path = "tests/temp/sim_srp.wav"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    sr = 16000
    input_signal = np.random.randn(48000)
    # t = np.arange(16000)
    # input_signal = np.sin(t*2*np.pi*1000/sr)
    
    sr, input_signal = wavfile.read("tests/fixtures/p225_001.wav")
    rt60 = None #0.3
    snr = 10
    n_grid_points = 50

    room_dims = np.array([6, 6, 3])
    source_coords = np.array([[2, 3, 1]])
    mic_coords = np.array([
        [0.5, 0.5, 1],
        [0.7, 5.5, 1],
        [5.5, 0.5, 1],
        [5.5, 5.2, 1]
    ])

    room = ConnectedShoeBox(room_dims, fs=sr, rt60=rt60)
    room.add_source(source_coords[0], input_signal)
    room.add_microphone_array(mic_coords)

    mic_signals = simulate(room, snr=snr)

    wavfile.write(temp_wav_path, sr, mic_signals.T)

    room_dims_batch = np.array(3*[room_dims])

    results = srp_phat(mic_signals[np.newaxis],
                       mic_coords[np.newaxis, :, :2],
                       room_dims[np.newaxis, :2],
                       sr, n_grid_points,
                       mode="interpolate",
                       phase_transform=True)
    grids = results["grid"]

    # fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    # print(grids.shape)
    for i in range(grids.shape[0]):
        plot_grid(grids[i], room_dims_batch[i], log=True,
                 microphone_coords=mic_coords, source_coords=source_coords)
    
    plt.savefig(temp_file_path)

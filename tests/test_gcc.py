import torch

import matplotlib.pyplot as plt
import numpy as np
import os
 
from scipy.io import wavfile
from sydra.sydra.dataset import SydraDataset

from ..pysoundloc.gcc import Gcc

from pyroomasync import ConnectedShoeBox, simulate


def test_gcc_on_sydra_dataset():
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = "tests/temp/gcc_sydra_{}.png"
    dataset = SydraDataset("tests/fixtures/1_source_anechoic")
    signals, metadata = dataset[2] # Analyzing dataset sample 2

    n_mics = signals.shape[1]

    mic_coords = metadata["mic_coordinates"][..., :2]
    room_dims = metadata["room_dims"][..., :2]
    source_coords = metadata["source_coordinates"][..., :2]
    sr = metadata["sr"]

    gcc_func = Gcc(sr)

    # Compute gcc for each microphone with the reference
    for i in range(1, n_mics):
        gcc_value, gcc_delays = gcc_func(signals[:, 0], signals[:, i])

        # TODO: compute ground truth delay

        plt.semilogx(gcc_delays, gcc_value[0])
    plt.savefig(temp_file_path)


def test_gcc_phat():
    os.makedirs("tests/temp", exist_ok=True)
    temp_file_path = "tests/temp/gcc.png"
    temp_wav_path = "tests/temp/sim_gcc.wav"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    sr = 16000
    input_signal = np.random.randn(48000)
    
    sr, input_signal = wavfile.read("tests/fixtures/p225_001.wav")
    rt60 = None # 0.3
    snr = None

    room_dims = np.array([6, 6, 3])
    source_coords = np.array([[2, 3, 1]])
    mic_coords = np.array([
        [0.5, 0.5, 1],
        [0.7, 5.5, 1],
    ])

    room = ConnectedShoeBox(room_dims, fs=sr, rt60=rt60)
    room.add_source(source_coords[0], input_signal)
    room.add_microphone_array(mic_coords)

    mic_signals = simulate(room, snr=snr)
    wavfile.write(temp_wav_path, sr, mic_signals.T)
    mic_signals = torch.from_numpy(mic_signals).unsqueeze(0)

    gcc_func = Gcc(sr)
    
    gcc = gcc_func(mic_signals[:, 0], mic_signals[:, 1])[0][0].numpy()

    plt.plot(gcc)
    plt.savefig(temp_file_path)

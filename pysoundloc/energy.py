import numpy as np

from .utils.math import compute_distance, compute_distance_grids


# Part 1: Measurements

def measure_mean_signal_energy(signal):
    energy = np.mean(signal**2)
    return energy


def measure_energy_ratio(signal_0, signal_1):
    energy_0 = measure_mean_signal_energy(signal_0)
    energy_1 = measure_mean_signal_energy(signal_1)
    return (energy_0/energy_1)**(-1/2)


def measure_energy_ratios(microphone_signals):
    n_mics = microphone_signals.shape[0]

    energy_ratios = {}
    for n_mic_0 in range(n_mics):
        for n_mic_1 in range(n_mics):#n_mic_0 + 1, n_mics):
            if n_mic_0 == n_mic_1:
                continue
            key = (n_mic_0, n_mic_1)
            
            energy_ratios[key] = measure_energy_ratio(microphone_signals[n_mic_0],
                                                      microphone_signals[n_mic_1])

    return energy_ratios


# Part 2: Theoretical metrics

def compute_theoretical_energy_ratios(microphone_coordinates, source_coordinates):
    n_mics = microphone_coordinates.shape[0]
    theoretical_tdoas = {}

    for n_mic_0 in range(n_mics):
        for n_mic_1 in range(n_mics):#n_mic_0 + 1, n_mics):
            if n_mic_0 == n_mic_1:
                continue
            key = (n_mic_0, n_mic_1)
            theoretical_tdoas[key] = compute_theoretical_energy_ratio(
                microphone_coordinates[n_mic_0],
                microphone_coordinates[n_mic_1],
                source_coordinates
            )

    return theoretical_tdoas


def compute_theoretical_energy_ratio_grids(mic_0_coords, mic_1_coords,
                                   room_dims, n_grid_points_per_axis=25):
    """Create a batch of grids where each point represents
    a position in a room with room_dims dimensions.
    The value of each cell is the attenuation ratio that a source leaving from the
    cell would have at one microphone and the other.
    """
    
    dist_0 = compute_distance_grids(mic_0_coords, room_dims,
                                    n_grid_points_per_axis)
    dist_1 = compute_distance_grids(mic_1_coords, room_dims,
                                    n_grid_points_per_axis)

    return (dist_0/dist_1)**2



def compute_theoretical_energy_ratio(mic_0_coords, mic_1_coords, source_coords):
    mic_0_coords = mic_0_coords[:2]
    mic_1_coords = mic_1_coords[:2]
    dist_0 = compute_distance(source_coords, mic_0_coords)
    dist_1 = compute_distance(source_coords, mic_1_coords)

    return (dist_0/dist_1)
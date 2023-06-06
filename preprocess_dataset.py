import matplotlib.pyplot as plt
import os

from tqdm import tqdm

from pysoundloc.models import srp_phat, music, tdoa_least_squares_ssl
from pysoundloc.visualization import plot_grid
from sydra.sydra.dataset import SydraDataset


def preprocess_dataset(method, input_dataset_dir, output_dataset_dir, n_grid_points=50,
                       plot_grids=True):
    """Apply a localization algorithm to an entire dataset

    Args:
        method (str): Name of the method to apply [srp | tdoa_ls | music]
        input_dataset_dir (str): Path to the input dataset in the SydraDataset format 
        output_dataset_dir (str): Path to the output dataset 
    """

    if method == "srp":
        method = srp_phat
    elif method == "music":
        method = music
    elif method == "tdoa_ls":
        method = tdoa_least_squares_ssl

    # TODO: Compute this in batch mode, using a dataloader
    dataset = SydraDataset(input_dataset_dir)

    os.makedirs(output_dataset_dir, exist_ok=True)

    results = []
    metadata_results = []
    for i, (signal, metadata) in tqdm(enumerate(dataset)):
        metadata_results.append(metadata)

        grid = srp_phat(
            signal, metadata["mic_coordinates"][..., :2],
            metadata["room_dims"][:2].unsqueeze(0), metadata["sr"],
            n_grid_points=n_grid_points
        )["grid"][0] # Unbatch it

        results.append(grid)
        
        if plot_grids:
            plot_grid(grid, metadata["room_dims"][:2], metadata["mic_coordinates"][0, :, :2],
                      source_coords=metadata["source_coordinates"][..., :2])
            plt.savefig(f"{output_dataset_dir}/{i}.png")


if __name__ == "__main__":
    preprocess_dataset(
        "srp", "/Users/ezajlerg/datasets/PP_GNN/2_sources_anechoic",
        "/Users/ezajlerg/datasets/PP_GNN/srp/2_sources_anechoic"
    )

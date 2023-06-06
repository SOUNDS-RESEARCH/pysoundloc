from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from .utils.math import compute_distance


def plot_correlations(correlations, output_path=None):
    for key, value in correlations.items():
        plt.plot(value[0], value[1], label=key)
    
    plt.legend()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_grid(error_grid, room_dims, microphone_coords=None, source_coords=None, log=False, ax=None,
                    num_ticks_per_axis=10, colorbar=True):
    if ax is None:
        fig, ax = plt.subplots()

    width, length = room_dims[:2]
    x_resolution = width/error_grid.shape[0]
    y_resolution = length/error_grid.shape[1]

    if log:
        norm = LogNorm()
        error_grid += 1e-7 # Prevent taking log(0)
    else:
        norm = None

    y_ticks = np.linspace(0, width, num_ticks_per_axis + 1)
    x_ticks = np.linspace(0, length, num_ticks_per_axis + 1)
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Length (m)")

    ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks_per_axis))
    ax.yaxis.set_major_locator(plt.MaxNLocator(num_ticks_per_axis))
    ax.xaxis.set_ticklabels(x_ticks)
    ax.yaxis.set_ticklabels(y_ticks)
    ax.set_xticklabels(["{:.1f}".format(x) for x in x_ticks])
    ax.set_yticklabels(["{:.1f}".format(y) for y in y_ticks])

    mesh = ax.pcolormesh(error_grid, cmap="YlGn", norm=norm)

    if colorbar:
        plt.colorbar(mesh, ax=ax)

    if microphone_coords is not None:
        mics_x = [mic[0]/x_resolution for mic in microphone_coords]
        mics_y = [mic[1]/y_resolution for mic in microphone_coords]
        ax.scatter(mics_y, mics_x, marker="o", label="microphones")

    if source_coords is not None:
        sources_x = source_coords[:, 0]*error_grid.shape[0]/room_dims[0]
        sources_y = source_coords[:, 1]*error_grid.shape[1]/room_dims[1]
        ax.scatter(sources_y, sources_x, marker="x", label="sources")

    if source_coords is not None or microphone_coords is not None:
        ax.legend()
    
    return mesh, ax


def plot_square_grid(grids, room_dims, mic_coords, source_coords, log=True):
    n_mics = mic_coords.shape[0]

    fig, axs = plt.subplots(nrows=n_mics, ncols=n_mics, figsize=(25, 20))
    for i, mics_ls_grids in enumerate(grids):
        for j, ls_grid in enumerate(mics_ls_grids):
            coords = mic_coords[[i, j]]
            plot_grid(ls_grid,
                      room_dims,
                      microphone_coords=coords,
                      source_coords=source_coords,
                      log=log, ax=axs[i,j])

            if i == 0:
                if i < n_mics:
                    axs[i,j].set_title(f"Mic {j}")
                else:
                    axs[i,j].set_title(f"Sum")
            if j == 0:
                axs[i,j].set_ylabel(f"Mic {i}")

    return axs


def get_2d_room_plot_axis(room, plot_mics_and_sources=True,
                          plot_distances=True):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, room.dims[0])
    plt.ylim(0, room.dims[1])

    if plot_mics_and_sources:
        mics = room.microphones.mic_array
        sources = room.sources.source_array
        ax = draw_mics_and_sources(ax, mics, sources)
    
    if plot_distances:
        _plot_source_to_microphone_distances(room, plt.gca())

    return ax


def plot_mics_and_sources(room_dims, mics, sources, ax=None, plot_distances=False):
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xlim(0, room_dims[0])
    ax.set_ylim(0, room_dims[1])

    draw_mics_and_sources(ax, mics, sources)
    
    if plot_distances:
        _plot_source_to_microphone_distances(ax, mics, sources)

    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Length (m)")
    return ax


def draw_mics_and_sources(ax, mics, sources):
    "Draw microphones and sources in an existing room"
    
    mics_x = [mic[0] for mic in mics]
    mics_y = [mic[1] for mic in mics]
    sources_x = [source[0] for source in sources]
    sources_y = [source[1] for source in sources]
    
    ax.scatter(mics_x, mics_y, marker="o", label="microphones")

    if len(sources_x) == 1:
        source_label = "source"
    else:
        source_label = "sources"

    ax.scatter(sources_x, sources_y, marker="x", label=source_label)
    # ax.legend()
    # ax.grid()

    return ax


def _plot_source_to_microphone_distances(ax, mics, sources):

    distance = compute_distance(mics[0], mics[1])
    ax.plot(
        [mics[0][0], mics[1][0]],
        [mics[0][1], mics[1][1]],
        "--", color="blue",
        label="distance={:.2f}m".format(distance)
    )
    for source in sources:
        for mic in mics:
            distance = compute_distance(source, mic)
            ax.plot(
                [source[0], mic[0]],
                [source[1], mic[1]],
                "--", color="grey",
                label="distance={:.2f}m".format(distance)
            )

    ax.legend()

    return ax

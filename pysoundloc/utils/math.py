import numpy as np
import torch

from .grid import create_fixed_size_grids

SPEED_OF_SOUND = 343.0


def compute_distance(p0, p1, mode="numpy"):
    "Compute the euclidean distance between two points"
    # n = min(p0.shape[0], p1.shape[0])
    # p0, p1 = p0[:n], p1[:n]

    if mode == "numpy":
        p0 = np.array(p0)
        p1 = np.array(p1)
        return np.linalg.norm(p0 - p1)
    elif mode == "torch":
        return torch.linalg.norm(p0 - p1)


def compute_distance_grids(reference_coords, room_dims,
                           n_grid_points_per_axis=25):
    """Create a grid of shape (n_grid_points_per_axis, n_grid_points_per_axis)
    Where each cell refers to a point in a room of room_dims.
    The value of each cell is the distance between that point and the reference point
    reference_coords. 

    This function is used to compute TDOA and energy grids.
    """

    coordinate_grids = create_fixed_size_grids(room_dims, n_grid_points_per_axis)[0]
    if isinstance(coordinate_grids, torch.Tensor):
        lib = torch
        norm = lambda x: torch.linalg.norm(x, dim=1)
    elif isinstance(coordinate_grids, np.ndarray):
        lib = np
        norm = lambda x: np.linalg.norm(x, axis=1)
    else:
        raise ValueError("'coordinate_grids' must be a numpy array or a torch tensor")

    if len(reference_coords.shape) == 1:
        # If a vector is provided, interpret it as the
        # the coordinates of a single reference (and not a batch)
        reference_coords = reference_coords.reshape((1,) + reference_coords.shape)
        coordinate_grids = coordinate_grids.reshape(
            (1,) + coordinate_grids.shape
        )

    reference_grids = lib.ones_like(coordinate_grids)

    reference_grids[:, :, :, :] = reference_coords[:, :, np.newaxis, np.newaxis]

    dist_grid = norm(reference_grids - coordinate_grids)
   
    return dist_grid


def normalize(x, min_x, max_x):
    return (x - min_x)/(max_x - min_x)


def denormalize(x, min_x, max_x):
    return x*(max_x - min_x) + min_x


def compute_doa_2_mics(m1, m2, s, radians=True):
    """Get the direction of arrival between two microphones and a source.
       The referential used is the direction of the two sources, that is,
       the vector m1 - m2.

       For more details, see: 
       https://math.stackexchange.com/questions/878785/how-to-find-an-angle-in-range0-360-between-2-vectors/879474

    Args:
        m1 (np.array): 2d coordinates of microphone 1
        m2 (np.array): 2d coordinates of microphone 2
        s (np.array): 2d coordinates of the source
        radians (bool): If True, result is between [-pi, pi). Else, result is between [0, 360)
    """

    reference_direction = m1 - m2
    mic_centre = (m1 + m2)/2
    source_direction = s - mic_centre

    doa = compute_angle_between_vectors(reference_direction, source_direction,
                                        radians=radians)

    return doa


def compute_angle_between_vectors(v1, v2, radians=True):
    dot = np.dot(v1, v2)
    det = np.linalg.det([v1, v2])

    doa = np.arctan2(det, dot)

    if not radians:
        doa = np.rad2deg(doa)
    
    return doa


def point_to_rect_min_distance(x, y, x_min, x_max, y_min, y_max):
    """Compute the minimum distance between a point (x, y) and a rectangle,
    defined by the points (x_min, y_min), (x_max, y_max).

    This function does not work for rotated rectangles.

    See: https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
    for a complete discussion
    """

    dist_x = _max_positive(x_min - x, x - x_max)
    dist_y = _max_positive(y_min - y, y - y_max)

    return (dist_x**2 + dist_y**2)**(1/2)


def grid_argmin(grids, grid_dims):
    """Estimate the location on a grid by picking the minimum value.
       'grid_dims' is a batch of tuples representing the length and width of the grid. 
    """
    is_tensor = isinstance(grids, torch.Tensor)
    if is_tensor: # Convert to numpy
        grids = grids.detach().cpu().numpy()
        grid_dims = grid_dims.detach().cpu().numpy()

    batch_size, n_points, _ = grids.shape
    
    estimated_locations = []

    # TODO: port to Pytorch
    for i in range(batch_size):
        # https://stackoverflow.com/questions/9482550/argmax-of-numpy-array-returning-non-flat-indices
        estimated_location_idx = np.unravel_index(grids[i].argmin(), grids[i].shape)
        x_coords = np.linspace(0, grid_dims[i, 0], n_points)
        y_coords = np.linspace(0, grid_dims[i, 1], n_points)
        estimated_location = [
            x_coords[estimated_location_idx[0]],
            y_coords[estimated_location_idx[1]]
        ]
        estimated_locations.append(estimated_location)

    if is_tensor:
        estimated_locations = torch.Tensor(estimated_locations)
    else:
        estimated_locations = np.array(estimated_locations)
         
    return estimated_locations


def grid_argmax(grids, grid_dims):
    """Estimate the location on a grid by picking the maximum value.
       'grid_dims' is a batch of tuples representing the length and width of the grid. 
    """

    return grid_argmin(-grids, grid_dims)


def _max_positive(x, y):
    "Variation of the 'max' function that the result is always non-negative"

    if isinstance(x, torch.Tensor):
        zeros = torch.zeros_like(x)
        return torch.stack([zeros, x, y]).max(dim=0)[0]
    elif isinstance(x, np.ndarray):
        zeros = np.zeros_like(x)
        return np.stack([zeros, x, y]).max(axis=0)
    else:
        return max(0, x, y)
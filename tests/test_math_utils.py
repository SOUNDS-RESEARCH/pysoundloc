import numpy as np

from ..pysoundloc.utils.math import (
    compute_doa_2_mics,
    point_to_rect_min_distance,
    point_to_rect_max_distance
)


def test_compute_doa_colinear():
    # m1 and m2 are on the x axis
    m1 = np.array([1, 0])
    m2 = np.array([0, 0])
    # s is also on the x axis
    s =  np.array([2, 0])
    
    doa_radians = compute_doa_2_mics(m1, m2, s)
    doa_degrees = compute_doa_2_mics(m1, m2, s, radians=False)
    assert doa_radians == 0
    assert doa_degrees == 0


def test_compute_doa_perpendicular():
    # m1 and m2 are on the x axis
    m1 = np.array([1, 0])
    m2 = np.array([-1, 0])

    # s is between the mics
    s =  np.array([0, 1])
    
    doa_degrees = compute_doa_2_mics(m1, m2, s, radians=False)
    doa_degrees_2 = compute_doa_2_mics(m1, m2, -s, radians=False)

    assert doa_degrees == 90
    assert doa_degrees_2 == -90


def test_point_to_rect_distance():

    x, y = (1.5, 1.8)

    x_min, y_min = (2.5, 2.7)
    x_max, y_max = (3.1, 4.2)
    
    dist = point_to_rect_min_distance(x, y, x_min, x_max, y_min, y_max)

    assert dist == 1.3453624047073711

def test_point_to_rect_distance():

    x, y = (1.5, 1.8)

    x_min, y_min = (2.5, 2.7)
    x_max, y_max = (3.1, 4.2)
    
    dist = point_to_rect_max_distance(x, y, x_min, x_max, y_min, y_max)

    assert dist == 2.8844410203711917

import numpy as np
import torch

from torch.nn import Module

from .utils.math import compute_distance


def compute_error(candidate, mic_1, mic_2, tdoa, norm="l1"):
    "Get a measure of how far a candidate point (x, y) is from a computed doa"

    dist_1 = compute_distance(candidate, mic_1)
    dist_2 = compute_distance(candidate, mic_2)

    error = tdoa - np.abs(dist_1 - dist_2)

    if norm == "l2":
        error = error**2
    elif norm == "l1":
        error = np.abs(error)
    
    return error


class SslMetrics(Module):
    def __init__(self, absolute_tolerance, relative_tolerance, dim=1, **kwargs):
        super().__init__()
    
        self.absolute_tolerance = AbsoluteToleranceScore(absolute_tolerance, dim=dim)
        self.room_relative_tolerance = RoomRelativeToleranceScore(relative_tolerance, dim=dim)
        self.room_relative_norm = RoomRelativeNorm(dim=dim)
        self.l2 = NormLoss("l2")
    
    def forward(self, model_output, targets, room_dims):
        
        targets = targets[:, 0, :2] # Single, planar source
        
        return {
            "absolute_tolerance": self.absolute_tolerance(model_output, targets),
            "room_relative_tolerance": self.room_relative_tolerance(model_output, targets, room_dims),
            "room_relative_norm": self.room_relative_norm(model_output, targets, room_dims),
            "l2": self.l2(model_output, targets)
        }


class AbsoluteToleranceScore(Module):
    def __init__(self, tolerance, norm_type="l2", dim=1):
        super().__init__()

        self.norm = NormLoss(norm_type, dim=dim)
        self.tolerance = tolerance
        self.dim = dim
    
    def forward(self, model_output, targets, mean_reduce=False):
        error = self.norm(model_output, targets) <= self.tolerance

        if mean_reduce:
            error = error.float().mean()

        return error


class RoomRelativeToleranceScore(Module):
    def __init__(self, tolerance, norm_type="l2", dim=1):
        super().__init__()

        self.norm = RoomRelativeNorm(norm_type, dim=dim)
        self.tolerance = tolerance
        self.dim = dim
    
    def forward(self, model_output, targets, room_dims, mean_reduce=False):
        error = self.norm(model_output, targets, room_dims, mean_reduce=False) <= self.tolerance
        if mean_reduce:
            error = error.float().mean()

        return error


class RoomRelativeNorm(Module):
    def __init__(self, norm_type="l2", dim=1):
        super().__init__()

        self.norm = NormLoss(norm_type, dim=dim)
        self.dim = dim
    
    def forward(self, model_output, targets, room_dims, mean_reduce=False):
        error = self.norm(model_output, targets)
        room_diagonals = torch.linalg.norm(room_dims, dim=self.dim)

        error /= room_diagonals

        if mean_reduce:
            error = error.mean()

        return error


class NormLoss(Module):
    def __init__(self, norm_type="l1", dim=1):
        super().__init__()

        if norm_type not in ["l1", "l2", "squared"]:
            raise ValueError("Supported norms are 'l1', 'l2', and 'squared'")
        self.norm_type = norm_type
        self.dim = dim

    def forward(self, model_output, targets, mean_reduce=False):
        error = model_output - targets
        if self.norm_type == "l1":
            normed_error = error.abs()
        elif self.norm_type == "l2" or self.norm_type == "squared":
            normed_error = error**2
        
        if self.dim is not None:
            normed_error = torch.sum(normed_error, dim=self.dim)

        if self.norm_type == "l2":
            normed_error = torch.sqrt(normed_error)

        if mean_reduce:
            normed_error = normed_error.mean()

        return normed_error

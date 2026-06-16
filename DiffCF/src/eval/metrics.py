import numpy as np


# ==========================================
# 1. Proximity Metrics (Distance Norms)
# ==========================================

def l1_distance_per_instance(x, y):
    """L1 Norm: Sum of absolute differences (Manhattan distance)"""
    diff = np.abs(x - y)
    return np.sum(diff, axis=(1, 2))


def l1_distance(x, y):
    return float(np.mean(l1_distance_per_instance(x, y)))


def l2_distance_per_instance(x, y):
    """L2 Norm: Euclidean distance"""
    diff = x - y
    return np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)


def l2_distance(x, y):
    return float(np.mean(l2_distance_per_instance(x, y)))


def l_inf_distance_per_instance(x, y):
    """L_inf Norm: Maximum single-point absolute difference (Chebyshev distance)"""
    diff = np.abs(x - y)
    return np.max(diff, axis=(1, 2))


def l_inf_distance(x, y):
    return float(np.mean(l_inf_distance_per_instance(x, y)))

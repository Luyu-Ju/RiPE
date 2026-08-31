"""Utility functions to process state."""
from typing import Tuple

import numpy as np
from numba import njit

# @jit
# def normalize(array_in):
#     """nx2 or mxnx2"""
#     if len(array_in.shape) == 2:
#         vec, fac = normalize_array(array_in)
#         return vec, fac
#     factors = []
#     vectors = []
#     for m in array_in:
#         vec, fac = normalize_array(m)
#         vectors.append(vec)
#         factors.append(fac)

#     return np.array(vectors), np.array(factors)


@njit
def vector_angles(vecs: np.ndarray) -> np.ndarray:
    """Calculate angles for an array of vectors
    :param vecs: nx2 ndarray
    :return: nx1 ndarray
    """
    ang = np.arctan2(vecs[:, 1], vecs[:, 0])  # atan2(y, x)
    return ang


@njit
def left_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([-1.0, 1.0])
    return vecs


@njit
def right_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([1.0, -1.0])
    return vecs

# @njit
def normalize(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize nx2 array along the second axis
    input: [n,2] ndarray
    output: (normalized vectors, norm factors)
    """
    norm_factors = np.linalg.norm(vecs, axis = 1)
    normalized = vecs / np.expand_dims(norm_factors, -1)
    # get rid of nans
    indices = np.where(norm_factors == 0)
    normalized[indices[0],:] = np.zeros(vecs.shape[1])
    return normalized, norm_factors


# @njit
def normalize3(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize nx2 array along the second axis
    input: [n,2] ndarray
    output: (normalized vectors, norm factors)
    """
    norm_factors = np.linalg.norm(vecs, axis = 2)
    norm_factors = np.array(norm_factors)
    normalized = vecs / np.expand_dims(norm_factors, -1)
    # get rid of nans
    indices = np.where(norm_factors == 0)
    normalized[indices[0],indices[1],:] = np.zeros(vecs.shape[2])
    return normalized, norm_factors
    
# @njit
def desired_directions(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 4:6] - state[:, 0:2]
    directions, dist = normalize(destination_vectors)
    return directions, dist


@njit
def vec_diff(vecs: np.ndarray) -> np.ndarray:
    """r_ab
    r_ab := r_a − r_b.
    """
    diff = np.expand_dims(vecs, 1) - np.expand_dims(vecs, 0)
    return diff


def each_diff(vecs: np.ndarray, keepdims=False) -> np.ndarray:
    """
    :param vecs: nx2 array
    :return: diff with diagonal elements removed
    """
    diff = vec_diff(vecs)
    # diff = diff[np.any(diff, axis=-1), :]  # get rid of zero vectors
    diff = diff[
        ~np.eye(diff.shape[0], dtype=bool), :
    ]  # get rif of diagonal elements in the diff matrix
    if keepdims:
        diff = diff.reshape(vecs.shape[0], -1, vecs.shape[1])

    return diff


# @njit
def speeds(state: np.ndarray) -> np.ndarray:
    """Return the speeds corresponding to a given state."""
    #     return np.linalg.norm(state[:, 2:4], axis=-1)
    speed_vecs = state[:, 2:4]
    speeds_array = np.array(np.linalg.norm(speed_vecs,axis = 1))
    return speeds_array


@njit
def center_of_mass(vecs: np.ndarray) -> np.ndarray:
    """Center-of-mass of a given group"""
    return np.sum(vecs, axis=0) / vecs.shape[0]


@njit
def minmax(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_min = np.min(vecs[:, 0])
    y_min = np.min(vecs[:, 1])
    x_max = np.max(vecs[:, 0])
    y_max = np.max(vecs[:, 1])
    return (x_min, y_min, x_max, y_max)

#@njit
def target(pos: np.ndarray, lasttarget: np.ndarray, coor: np.ndarray, guidelineIndex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    ## calculate angle between vector to last target and vector to potential targets
    # calculate x y distance between each pedestrians and each targets
    lasttarget = lasttarget.astype(int)
    newtarget = np.zeros(len(lasttarget))
    newtarget = newtarget.astype(int)
    lasttargetcoor = np.zeros((len(lasttarget),2))
    for z in range(len(coor)):
        id = np.where(guidelineIndex == z+1)
        posi = pos[id[0],:]
        coori = coor[z]['coor']
        
        # diff = np.expand_dims(posi, axis=0) - coori[:, np.newaxis]
        # diff_T = np.transpose(diff, (1, 0, 2))
        # calculate x y distance between each pedestrians and its last target
        lasttargeti = lasttarget[id[0]]
        lasttargetcoori = coori[lasttargeti.flatten(), :]
        lasttarget_vec = posi - lasttargetcoori
        # lasttarget_vec = np.expand_dims(lasttarget_vec, axis=1)
        # expand_lasttarget = np.repeat(lasttarget_vec, diff_T.shape[1], axis=1)
        # # calculate angle between vector to last target and vector to potential targets
        # angles = np.degrees(np.arccos(np.sum(diff_T * expand_lasttarget, axis=2) / (np.linalg.norm(diff_T, axis=2) * np.linalg.norm(expand_lasttarget, axis=2))))
        
        # # judge wheter change to new target
        # nexttarget = np.minimum(lasttargeti + 1, len(coori)-1)
        # angle_next = angles[np.arange(lasttargeti.shape[0])[:, np.newaxis], nexttarget]
        # indicesangle = np.where(angle_next >= 90)[0]
        # lasttargeti[indicesangle] += 1
        # lasttargeti[lasttargeti > len(coori)-1] = len(coori)-1
        # newtarget[id] = lasttargeti
        # lasttargetcoor[id,:] = lasttargetcoori
        distance = np.linalg.norm(lasttarget_vec, axis=1)
        indicesdist = np.where(distance <= 1.5)
        lasttargeti[indicesdist[0]] += 1
        lasttargeti[lasttargeti > len(coori)-1] = len(coori)-1
        newtarget[id[0]] = lasttargeti.reshape(-1)
        lasttargetcoor[id[0],:] = lasttargetcoori
    # targetnewcoor = coor[vec2.flatten(), :]
    # vector = newtarget-lasttarget.reshape(-1)
    # count_zeros = np.count_nonzero(vector == 0)
    # print(count_zeros)
    return newtarget, lasttargetcoor
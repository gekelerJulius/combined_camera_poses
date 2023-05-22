from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from classes.logger import Divider


def icp(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The Iterative Closest Point method. Finds the best-fit transform that maps points A on to points B.
    :param A: (N, 3) numpy ndarray.
    :param B: (N, 3) numpy ndarray.
    :return: R: (3, 3) numpy ndarray, distances: (N,) numpy ndarray.
    """
    assert len(A) == len(B)
    centroid_A: np.ndarray = np.mean(A, axis=0)
    centroid_B: np.ndarray = np.mean(B, axis=0)

    Am: np.ndarray = A - centroid_A
    Bm: np.ndarray = B - centroid_B

    # rotation matrix
    H: np.ndarray = np.dot(Am.T, Bm)

    # singular value decomposition
    U, S, Vt = np.linalg.svd(H)
    R: np.ndarray = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t: np.ndarray = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T: np.ndarray = np.identity(A.shape[1] + 1)
    T[:A.shape[1], :A.shape[1]] = R
    T[:A.shape[1], A.shape[1]] = t

    # Apply the transformation to the points in A
    A_transformed: np.ndarray = np.dot(R, A.T) + t.reshape(-1, 1)

    # Find the nearest neighbors in B
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(B)
    distances, indices = nbrs.kneighbors(A_transformed.T)
    return T, distances.ravel()


def test_icp():
    R = Rotation.random().as_matrix()
    A = np.random.rand(500, 3)
    B = np.dot(R, A.T).T

    T, distances = icp(A, B)
    T = np.array(T)
    tolerance = 0.1

    # In same number format
    print("T")
    print(T)
    print("R")
    print(R)
    print("T - R")
    print(T[:3, :3] - R)
    with Divider():
        print("Mean, std, max, min")
        print(np.mean(distances))
        print(np.std(distances))
        print(np.max(distances))
        print(np.min(distances))
    assert np.allclose(distances, 0, atol=tolerance)

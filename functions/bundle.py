import numpy as np
from numpy import ndarray
from scipy.optimize import least_squares
from typing import Tuple

from scipy.spatial.transform import Rotation


def bundle_adjustment_sparsity(n_cameras: int, n_points: int, camera_indices: ndarray,
                               point_indices: ndarray) -> ndarray:
    assert camera_indices.size == n_cameras, "camera_indices must have the same size as n_cameras"
    assert point_indices.size == n_points, "point_indices must have the same size as n_points"
    assert camera_indices.min() >= 0, "camera_indices must be non-negative"
    assert point_indices.min() >= 0, "point_indices must be non-negative"
    assert camera_indices.max() < n_cameras, "camera_indices must be less than n_cameras"
    assert point_indices.max() < n_points, "point_indices must be less than n_points"

    m: int = 2 * n_points
    n: int = n_cameras * 6 + n_points * 3
    A: ndarray = np.zeros((m, n), dtype=int)

    for i in range(n_points):
        for j in range(n_cameras):
            point_index: int = point_indices[i]
            camera_index: int = camera_indices[j]
            A[2 * i, camera_index * 6: (camera_index + 1) * 6] = 1
            A[2 * i + 1, camera_index * 6: (camera_index + 1) * 6] = 1
            A[2 * i, n_cameras * 6 + point_index * 3: n_cameras * 6 + (point_index + 1) * 3] = 1
            A[2 * i + 1, n_cameras * 6 + point_index * 3: n_cameras * 6 + (point_index + 1) * 3] = 1
    return A


def project(points: ndarray, R: ndarray, T: ndarray, K: ndarray) -> ndarray:
    assert points.shape[1] == 3, "points must have shape (N, 3)"
    assert R.shape == (3, 3), "R must have shape (3, 3)"
    assert T.shape == (3,), "T must have shape (3,)"
    assert K.shape == (3, 3), "K must have shape (3, 3)"

    camera_params = np.hstack((R, T))
    points_proj = points.dot(camera_params[:3, :3].T) + camera_params[:3, 3]
    points_proj = points_proj.dot(K.T)
    points_proj /= points_proj[:, 2, np.newaxis]
    return points_proj[:, :2]


def fun(params: ndarray, n_cameras: int, n_points: int, camera_indices: ndarray, point_indices: ndarray,
        points_2d: ndarray, K1: ndarray, K2: ndarray) -> ndarray:
    assert params.size == n_cameras * 6 + n_points * 3, f"params size does not match n_cameras and n_points: {params.size} != {n_cameras * 6 + n_points * 3}"
    assert camera_indices.size == n_cameras, f"camera_indices must have the same size as n_cameras {camera_indices.size} != {n_cameras}"
    assert point_indices.size == n_points, f"point_indices must have the same size as n_points {point_indices.size} != {n_points}"
    assert points_2d.shape[
               0] == 2 * n_points, f"points_2d must have shape (2 * N, 2) {points_2d.shape[0]} != {2 * n_points}"
    assert K1.shape == (3, 3), f"K1 must have shape (3, 3) {K1.shape}"
    assert K2.shape == (3, 3), f"K2 must have shape (3, 3) {K2.shape}"

    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices, :3],
                          camera_params[camera_indices, 3:], K1)
    return (points_proj - points_2d).ravel()


def do_bundle(points1: ndarray, points2: ndarray, K1: ndarray, K2: ndarray, R: ndarray, T: ndarray) -> \
        Tuple[ndarray, ndarray, ndarray]:
    assert points1.shape == points2.shape, "points1 and points2 must have the same shape"
    assert K1.shape == K2.shape, "K1 and K2 must have the same shape"
    assert R.shape == (3, 3), "R must have shape (3, 3)"
    assert T.shape == (3,), "T must have shape (3,)"

    rot: Rotation = Rotation.from_matrix(R)
    n_cameras = 2
    n_points = len(points1)
    camera_indices = np.array([0, 1])
    point_indices = np.array(range(n_points))
    points_2d = np.concatenate((points1, points2), axis=0)

    x0 = np.hstack((np.concatenate((R.ravel(), T.ravel(), np.zeros(6))),
                    np.random.randn(n_points // 2 * 3)))
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    assert A.shape == (2 * n_points, n_cameras * 6 + n_points * 3), \
        f"A has wrong shape {A.shape}. Expected {(2 * n_points, n_cameras * 6 + n_points * 3)}"

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K1, K2))
    params_optimized = res.x
    camera_params_optimized = params_optimized[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d_optimized = params_optimized[n_cameras * 6:].reshape((n_points, 3))
    P1 = K1 @ np.hstack((camera_params_optimized[0, :3].reshape(3, 1), camera_params_optimized[0, 3:]))
    P2 = K2 @ np.hstack((camera_params_optimized[1, :3].reshape(3, 1), camera_params_optimized[1, 3:]))
    return P1, P2, points_3d_optimized

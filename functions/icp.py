from typing import Tuple

import numpy as np
from open3d.open3d_pybind.geometry import PointCloud
from open3d.open3d_pybind.registration import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from classes.logger import Divider
import open3d as o3d


def preprocess_point_cloud(pcd, voxel_size):
    """
    Downsample the point cloud, estimate normals, then compute a FPFH feature for each point.
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down: PointCloud, target_down: PointCloud, source_fpfh, target_fpfh, voxel_size):
    """
    Use RANSAC for global registration.
    """
    distance_threshold = voxel_size * 1.5
    result = registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        distance_threshold,
        TransformationEstimationPointToPoint(False), 4,
        [CorrespondenceCheckerBasedOnEdgeLength(0.9),
         CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, transformation):
    """
    Refine the registration using ICP.
    """
    print("Refine ICP registration...")
    distance_threshold = 0.02
    result = registration_icp(
        source, target, distance_threshold, transformation,
        TransformationEstimationPointToPoint())
    return result


def test_registration():
    # generate some clean (noise-free) point cloud data
    N = 1000
    src_pts = np.random.rand(N, 3)

    # apply a known rotation and translation
    R_true = Rotation.from_euler('zyx', [45, 45, 30], degrees=True)
    t_true = np.array([1, 1, 1])

    dst_pts = R_true.apply(src_pts) + t_true

    # create open3d point cloud objects
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    source.points = o3d.utility.Vector3dVector(src_pts)
    target.points = o3d.utility.Vector3dVector(dst_pts)

    # apply global registration
    voxel_size = 0.05  # size of the voxels used for down-sampling and normal estimation
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("Initial alignment")
    print(result_ransac)

    # refine with ICP
    result_icp = refine_registration(source, target, result_ransac.transformation)
    print("Final Transformation:")
    print(result_icp.transformation)
    assert np.allclose(result_icp.transformation, np.hstack((R_true.as_matrix(), t_true.reshape(-1, 1))), atol=0.1)


def icp(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The Iterative Closest Point method. Finds the best-fit transform that maps points A on to points B.
    :param A: (N, 3) numpy ndarray.
    :param B: (N, 3) numpy ndarray.
    :return: P: (4, 4) numpy ndarray, distances: (N,) numpy ndarray.
    """
    A = np.array(A)
    B = np.array(B)
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

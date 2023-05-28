from typing import Tuple

import numpy as np
from numpy import ndarray
from open3d.cpu.pybind.geometry import *
from open3d.cpu.pybind.pipelines.registration import *
from scipy.spatial.transform import Rotation
import open3d as o3d
import logging

# Configure logging to suppress Open3D output
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def do_icp(
        pts1: ndarray, pts2: ndarray, corresponding_by_index=False
) -> RegistrationResult:
    """
    The Iterative Closest Point method. Finds the best-fit transform that maps points A on to points B.
    :param pts1: (N, 3) numpy ndarray.
    :param pts2: (N, 3) numpy ndarray.
    :param corresponding_by_index: If True, the points are assumed to be in correspondence by index.
    :return: P: (4, 4) numpy ndarray, distances: (N,) numpy ndarray.
    """
    pts1 = np.copy(pts1)
    pts2 = np.copy(pts2)
    assert (
            pts1.shape == pts2.shape and pts1.shape[1] == 3
    ), "pts1 and pts2 must be (N, 3) ndarrays."

    cloud1: PointCloud = o3d.geometry.PointCloud()
    cloud1.points = o3d.utility.Vector3dVector(pts1)
    cloud2: PointCloud = o3d.geometry.PointCloud()
    cloud2.points = o3d.utility.Vector3dVector(pts2)
    voxel_size = 0.05

    if corresponding_by_index:
        correspondences = np.array([[i, i] for i in range(pts1.shape[0])])
        corr_set = o3d.utility.Vector2iVector(correspondences)
        init_transformation = (
            TransformationEstimationPointToPoint().compute_transformation(
                cloud1, cloud2, corr_set
            )
        )
    else:
        cloud1, cloud1_fpfh = preprocess_point_cloud(cloud1, voxel_size)
        cloud2, cloud2_fpfh = preprocess_point_cloud(cloud2, voxel_size)
        result_init: RegistrationResult = execute_global_registration(
            cloud1, cloud2, cloud1_fpfh, cloud2_fpfh, voxel_size
        )
        init_transformation = result_init.transformation

    result_icp: RegistrationResult = refine_registration(
        cloud1, cloud2, init_transformation
    )
    # draw_registration_result(cloud1, cloud2, result_icp.transformation, as_pose=True)
    return result_icp


def preprocess_point_cloud(
        pcd: PointCloud, voxel_size: float
) -> Tuple[PointCloud, ndarray]:
    """
    Downsample the point cloud, estimate normals, then compute a FPFH feature for each point.
    """
    logging.basicConfig(
        level=logging.NOTSET,
    )
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    pcd_fpfh = compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
        source_down: PointCloud,
        target_down: PointCloud,
        source_fpfh,
        target_fpfh,
        voxel_size,
) -> RegistrationResult:
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
        TransformationEstimationPointToPoint(False),
        4,
        [
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        RANSACConvergenceCriteria(4000000, 500),
    )
    return result


def refine_registration(source, target, transformation) -> RegistrationResult:
    distance_threshold = 0.02
    result = registration_icp(
        source,
        target,
        distance_threshold,
        transformation,
        TransformationEstimationPointToPoint(),
    )
    return result


def test_registration():
    # generate some clean (noise-free) point cloud data
    N = 1000
    src_pts = np.random.rand(N, 3)

    # apply a known rotation and translation
    random_angles: ndarray = np.random.rand(3) * 90
    print("True Rotation:")
    print(random_angles)
    R_true = Rotation.from_euler("zyx", random_angles, degrees=True)
    random_translation: ndarray = np.random.rand(3) * 10
    print("True Translation:")
    print(random_translation)
    t_true = np.array(random_translation)

    dst_pts = R_true.apply(src_pts) + t_true

    # Add noise
    noise = np.random.normal(0, 0.01, (N, 3))
    dst_pts += noise

    # create open3d point cloud objects
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()

    source.points = o3d.utility.Vector3dVector(src_pts)
    target.points = o3d.utility.Vector3dVector(dst_pts)

    # apply global registration
    voxel_size = 0.05  # size of the voxels used for down-sampling and normal estimation
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    result_ransac = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size
    )
    print("Initial alignment")
    print(result_ransac)

    # refine with ICP
    result_icp = refine_registration(source, target, result_ransac.transformation)
    draw_registration_result(source, target, result_icp.transformation)
    print("Final Rotation:")
    print(
        Rotation.from_matrix(np.copy(result_icp.transformation)[:3, :3]).as_euler(
            "zyx", degrees=True
        )
    )
    print("Final Translation:")
    print(np.copy(result_icp.transformation)[:3, 3])
    return 0


def draw_registration_result(
        cloud1: PointCloud, cloud2: PointCloud, transformation: ndarray, as_pose=False
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    max1 = np.max(cloud1.get_max_bound())
    max2 = np.max(cloud2.get_max_bound())
    max = np.max([max1, max2])
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=max))

    # if as_pose:
    #     conns = CONNECTIONS_LIST
    #     for i, conn in enumerate(conns):
    #         vis.add_geometry(o3d.geometry.LineSet(
    #             points=o3d.utility.Vector3dVector([copy.points[conn[0]], copy.points[conn[1]]]),
    #             lines=o3d.utility.Vector2iVector([[0, 1]])))
    # else:
    vis.add_geometry(cloud1)
    cloud1.paint_uniform_color([0, 0, 1])
    vis.add_geometry(cloud2)
    cloud2.paint_uniform_color([0, 1, 0])
    vis.run()

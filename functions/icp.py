from typing import Tuple

import numpy as np
from numpy import ndarray
from open3d.cpu.pybind.geometry import *
from open3d.cpu.pybind.pipelines.registration import *
from open3d.cpu.pybind.utility import Vector3dVector
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
    :param pts1: (N, 3) or (N, 6) numpy ndarray, where the first 3 columns are
                 the xyz coordinates and the last 3 columns are the rgb values.
    :param pts2: Same as pts1.
    :param corresponding_by_index: If True, the points are assumed to be in correspondence by index.
    """
    pts1 = np.copy(pts1)
    pts2 = np.copy(pts2)
    assert (
        pts1.shape == pts2.shape and pts1.shape[1] == 3 or pts1.shape[1] == 6
    ), "pts1 and pts2 must be (N, 3) or (N, 6) ndarrays."

    cloud1 = PointCloud()
    cloud1.points = Vector3dVector(pts1[:, :3])
    cloud2 = PointCloud()
    cloud2.points = Vector3dVector(pts2[:, :3])
    voxel_size = 0.2
    result_init: RegistrationResult
    result_icp: RegistrationResult

    if corresponding_by_index:
        correspondences = np.array([[i, i] for i in range(pts1.shape[0])])
        corr_set = o3d.utility.Vector2iVector(correspondences)
        init_transformation = (
            TransformationEstimationPointToPoint().compute_transformation(
                cloud1, cloud2, corr_set
            )
        )
        # result_init = registration_fgr_based_on_correspondence(
        #     cloud1, cloud2, corr_set, FastGlobalRegistrationOption()
        # )
        # init_transformation = np.asarray(result_init.transformation)
    else:
        cloud1, cloud1_fpfh = preprocess_point_cloud(cloud1, voxel_size)
        cloud2, cloud2_fpfh = preprocess_point_cloud(cloud2, voxel_size)
        result_init = execute_global_registration(
            cloud1, cloud2, cloud1_fpfh, cloud2_fpfh, voxel_size
        )
        init_transformation = np.asarray(result_init.transformation)

    if pts1.shape[1] == 6 and pts2.shape[1] == 6:
        cloud1.colors = Vector3dVector(pts1[:, 3:6])
        cloud2.colors = Vector3dVector(pts2[:, 3:6])
        # result_icp = colored_registration(cloud1, cloud2, init_transformation)
        result_icp = refine_registration(
            cloud1, cloud2, init_transformation, voxel_size * 0.4
        )
    else:
        result_icp = refine_registration(
            cloud1, cloud2, init_transformation, voxel_size * 0.4
        )

    # draw_registration_result(cloud1, cloud2, result_icp.transformation, as_pose=True)
    return result_icp


def do_icp_correl(pts1: ndarray, pts2: ndarray) -> Tuple[ndarray, float]:
    """
    The Iterative Closest Point method. Finds the best-fit transform that maps points A on to points B.
    :param pts1: (N, 3) or (N, 6) numpy ndarray, where the first 3 columns are
                 the xyz coordinates and the last 3 columns are the rgb values.
    :param pts2: Same as pts1.
    :return: P: (4, 4) numpy ndarray
    """
    pts1 = np.copy(pts1)
    pts2 = np.copy(pts2)
    assert (
        pts1.shape == pts2.shape and pts1.shape[1] == 3
    ), "pts1 and pts2 must be (N, 3) or (N, 6) ndarrays."

    cloud1 = PointCloud()
    cloud1.points = Vector3dVector(pts1)
    cloud2 = PointCloud()
    cloud2.points = Vector3dVector(pts2)
    correspondences = np.array([[i, i] for i in range(pts1.shape[0])])
    corr_set = o3d.utility.Vector2iVector(correspondences)
    transformation = TransformationEstimationPointToPoint().compute_transformation(
        cloud1, cloud2, corr_set
    )

    pts1cloned = np.copy(pts1)
    pts1cloned = np.hstack((pts1cloned, np.ones((pts1cloned.shape[0], 1))))
    pts1cloned = np.dot(transformation, pts1cloned.T).T
    pts1cloned = pts1cloned[:, :3]
    cloned: PointCloud = PointCloud()
    cloned.points = Vector3dVector(pts1cloned)
    rmse = TransformationEstimationPointToPoint().compute_rmse(cloned, cloud2, corr_set)
    return transformation, rmse


def calculate_rmse(pts1: ndarray, pts2: ndarray) -> float:
    """
    :param pts1: (N, 3) numpy ndarray, where the 3 columns are
                 the xyz coordinates
    :param pts2: Same as pts1.
    :return: rmse: float
    """
    pts1 = np.copy(pts1)
    pts2 = np.copy(pts2)
    assert (
        pts1.shape == pts2.shape and pts1.shape[1] == 3
    ), "pts1 and pts2 must be (N, 3) ndarrays."

    cloud1 = PointCloud()
    cloud1.points = Vector3dVector(pts1)
    cloud2 = PointCloud()
    cloud2.points = Vector3dVector(pts2)
    correspondences = np.array([[i, i] for i in range(pts1.shape[0])])
    corr_set = o3d.utility.Vector2iVector(correspondences)
    return TransformationEstimationPointToPoint().compute_rmse(cloud1, cloud2, corr_set)


# CURRENTLY NOT WORKING
def colored_registration(
    source: PointCloud, target: PointCloud, initial_transformation: ndarray
) -> RegistrationResult:
    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [0.4, 0.2, 0.1]
    max_iter = [1000, 600, 300]
    current_transformation = np.identity(4)
    result_icp = None
    print("3. Colored point cloud registration")
    for scale in range(3):
        cur_iter = max_iter[scale]
        radius = voxel_radius[scale]
        # print([cur_iter, radius, scale])
        #
        # print("3-1. Downsample with a voxel size %.2f" % radius)
        # source_down = source.voxel_down_sample(radius)
        # target_down = target.voxel_down_sample(radius)
        #
        # print("3-2. Estimate normal.")
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        )
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        )

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source=source,
            target=target,
            max_correspondence_distance=radius,
            init=current_transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-2, relative_rmse=1e-2, max_iteration=cur_iter
            ),
        )
        current_transformation = result_icp.transformation
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


def refine_registration(
    source, target, transformation, voxel_size
) -> RegistrationResult:
    result = registration_icp(
        source,
        target,
        voxel_size,
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

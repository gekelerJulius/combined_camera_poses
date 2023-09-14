from typing import Optional, Tuple

import cv2
import numpy as np
from numpy import ndarray
from functions.bundle import do_bundle


def refine_extrinsic_estimation(
    estimated_extrinsic: Optional[ndarray],
    points1_img: ndarray,
    points2_img: ndarray,
    K1: ndarray,
    K2: ndarray,
) -> Optional[Tuple[ndarray, ndarray, float]]:
    """
    Refines the extrinsic estimation by using bundle adjustment.
    Args:
        estimated_extrinsic: The extrinsic estimation to refine. From Camera 1 to Camera 2.
        points1_img: New image points from camera 1
        points2_img: New image points from camera 2
        K1: Intrinsic matrix of camera 1
        K2: Intrinsic matrix of camera 2
    Returns: Updated extrinsic estimation
    """
    points1_img = np.array(points1_img)
    points2_img = np.array(points2_img)
    assert points1_img.shape == points2_img.shape
    assert points1_img.shape[1] == 2
    if points1_img.shape[0] < 8:
        return None

    if estimated_extrinsic is None:
        E = np.zeros((3, 3))
        R = np.zeros((3, 3))
        t = np.zeros((3, 1))
        mask = np.zeros((len(points1_img), 1))
        cv2.recoverPose(
            points1=points1_img,
            points2=points2_img,
            cameraMatrix1=K1,
            cameraMatrix2=K2,
            distCoeffs1=None,
            distCoeffs2=None,
            method=cv2.RANSAC,
            prob=0.99999,
            threshold=1,
            E=E,
            R=R,
            t=t,
            mask=mask,
        )
        estimated_extrinsic = np.hstack((R, t))
    assert estimated_extrinsic.shape == (3, 4)

    # R = estimated_extrinsic[:, :3]
    # t = estimated_extrinsic[:, 3]
    # E1, E2, points_3d, error = do_bundle(
    #     points1_img, points2_img, K1, K2, R, t.reshape((3,))
    # )

    # ///////////////////////////////
    E = np.zeros((3, 3))
    R = np.zeros((3, 3))
    t = np.zeros((3, 1))
    mask = np.zeros((len(points1_img), 1))
    cv2.recoverPose(
        points1=points1_img,
        points2=points2_img,
        cameraMatrix1=K1,
        cameraMatrix2=K2,
        distCoeffs1=None,
        distCoeffs2=None,
        method=cv2.RANSAC,
        prob=0.99999,
        threshold=1,
        E=E,
        R=R,
        t=t,
        mask=mask,
    )
    E2 = np.hstack((R, t))
    points_3d = np.zeros((points1_img.shape[0], 3))
    error = 0
    # ///////////////////////////////

    # Plot points 3d
    # fig: Figure = PlotService.get_instance().get_plot("3d points") if PlotService.get_instance().plot_exists(
    #     "3d points") else plt.figure()
    # ax: Axes3D = fig.add_subplot(111, projection='3d')
    # ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # PlotService.get_instance().add_plot(fig, "3d points")
    # plt.pause(20)
    return E2, points_3d, error

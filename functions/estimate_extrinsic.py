from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray
import cv2 as cv

from classes.PlotService import PlotService
from classes.logger import Divider
from functions.bundle import do_bundle


def estimate_extrinsic(
        points1_img: ndarray,
        points2_img: ndarray,
        K1: ndarray,
        K2: ndarray,
) -> Optional[ndarray]:
    assert points1_img.shape == points2_img.shape
    assert points1_img.shape[1] == 2
    if points1_img.shape[0] < 8:
        return None

    points1_img = np.array(points1_img)
    points2_img = np.array(points2_img)
    mask = np.zeros((len(points1_img), 1))

    # //////////////////////////////////////
    # F, mask = cv.findFundamentalMat(
    #     points1=points1_img,
    #     points2=points2_img,
    #     method=cv.FM_RANSAC,
    #     ransacReprojThreshold=1,
    #     confidence=0.99999,
    #     maxIters=10000,
    #     mask=mask,
    # )
    # E = K2.T @ F @ K1
    # R1, R2, t = cv.decomposeEssentialMat(E)
    # P1 = np.hstack((R1, t))
    # P2 = np.hstack((R1, -t))
    # P3 = np.hstack((R2, t))
    # P4 = np.hstack((R2, -t))
    # possible_projections = [P1, P2, P3, P4]
    # identity_proj = np.hstack((np.eye(3), np.zeros((3, 1))))
    # P = None
    # min_cheirality_error = np.inf
    # for projection in possible_projections:
    #     points4d = cv.triangulatePoints(
    #         projMatr1=identity_proj,
    #         projMatr2=projection,
    #         projPoints1=points1_img.T,
    #         projPoints2=points2_img.T,
    #     )
    #     points3d = points4d[:3, :] / points4d[3, :]
    #     points3d = points3d.T
    #
    #     cheirality_error = 0
    #     for point in points3d:
    #         if point[2] < 0:
    #             cheirality_error += 1
    #
    #     cheirality_error /= len(points3d)
    #     if cheirality_error < min_cheirality_error:
    #         min_cheirality_error = cheirality_error
    #         P = projection
    #
    # if P is not None:
    #     # with Divider("P"):
    #     #     print(P)
    #     #     print(f"with cheirality error {min_cheirality_error}")
    #     return P
    # //////////////////////////////////////

    E = np.zeros((3, 3))
    R = np.zeros((3, 3))
    t = np.zeros((3, 1))
    mask = np.zeros((len(points1_img), 1))
    cv.recoverPose(
        points1=points1_img,
        points2=points2_img,
        cameraMatrix1=K1,
        cameraMatrix2=K2,
        distCoeffs1=None,
        distCoeffs2=None,
        method=cv.RANSAC,
        prob=0.99999,
        threshold=1,
        E=E,
        R=R,
        t=t,
        mask=mask,
    )
    proj_matrix = np.hstack((R, t))
    P1, P2, points_3d = do_bundle(points1_img, points2_img, K1, K2, R, t.reshape((3,)))

    with Divider("P1"):
        print(P1)
    with Divider("P2"):
        print(P2)
    # Plot points 3d
    fig: Figure = PlotService.get_instance().get_plot("3d points") if PlotService.get_instance().plot_exists(
        "3d points") else plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    PlotService.get_instance().add_plot(fig, "3d points")
    plt.pause(20)
    return proj_matrix

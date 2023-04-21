import os
import cv2 as cv

import numpy as np
from matplotlib import pyplot as plt

from functions.funcs import (
    extract_extrinsics_from_xml,
    extract_intrinsics_from_xml,
    match_features,
)

wildtrack_path = "E:\\Uni\\Bachelor\\Wildtrack_dataset"

# Path to the folder containing the images
images_path = os.path.join(wildtrack_path, "Image_subsets")
camera1_path = os.path.join(images_path, "C1")
camera2_path = os.path.join(images_path, "C2")
camera3_path = os.path.join(images_path, "C3")
camera4_path = os.path.join(images_path, "C4")
camera5_path = os.path.join(images_path, "C5")
camera6_path = os.path.join(images_path, "C6")
camera7_path = os.path.join(images_path, "C7")

# Path to the folder containing the annotations
annotations_path = os.path.join(wildtrack_path, "annotations_positions")

# Path to the folder containing the calibrations
calibrations_path = os.path.join(wildtrack_path, "calibrations")

extrinsics_paths = [
    os.path.join(calibrations_path, "extrinsic", f"extr_CVLab{i}.xml")
    for i in range(1, 8)
]

intrinsics_paths = [
    os.path.join(calibrations_path, "intrinsic_zero", f"intr_CVLab{i}.xml")
    for i in range(1, 8)
]

rvs = []
tvs = []
camera_matrices = []
dist_coeffs = []

for i in range(1, 8):
    rv, tv = extract_extrinsics_from_xml(extrinsics_paths[i - 1])
    camera_matrix, dist_coeff = extract_intrinsics_from_xml(intrinsics_paths[i - 1])

    rvs.append(rv)
    tvs.append(tv)
    camera_matrices.append(camera_matrix)
    dist_coeffs.append(dist_coeff)

first_images = [
    os.path.join(camera1_path, "00000000.png"),
    os.path.join(camera2_path, "00000000.png"),
    os.path.join(camera3_path, "00000000.png"),
    os.path.join(camera4_path, "00000000.png"),
    os.path.join(camera5_path, "00000000.png"),
    os.path.join(camera6_path, "00000000.png"),
    os.path.join(camera7_path, "00000000.png"),
]

undistorted_images = []
for i in range(7):
    image = cv.imread(first_images[i])
    h, w = image.shape[:2]

    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
        camera_matrices[i], dist_coeffs[i], (w, h), 1, (w, h)
    )

    # undistort
    dst = cv.undistort(
        image, camera_matrices[i], dist_coeffs[i], None, new_camera_matrix
    )

    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    undistorted_images.append(dst)

# Find matches between undistorted images from camera 1 and camera 2
pts1, pts2 = match_features(undistorted_images[0], undistorted_images[1])

# Find the fundamental matrix
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

# Keep only inliers
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Rectify the images
ret1, ret2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    camera_matrices[0],
    dist_coeffs[0],
    camera_matrices[1],
    dist_coeffs[1],
    undistorted_images[0].shape[:2],
    rvs[1],
    tvs[1],
    flags=cv.CALIB_ZERO_DISPARITY,
    alpha=-1,
)

map1x, map1y = cv.initUndistortRectifyMap(
    camera_matrices[0],
    dist_coeffs[0],
    ret1,
    P1,
    undistorted_images[0].shape[:2],
    cv.CV_32FC1,
)
map2x, map2y = cv.initUndistortRectifyMap(
    camera_matrices[1],
    dist_coeffs[1],
    ret2,
    P2,
    undistorted_images[1].shape[:2],
    cv.CV_32FC1,
)

rectified1 = cv.remap(undistorted_images[0], map1x, map1y, cv.INTER_LINEAR)
rectified2 = cv.remap(undistorted_images[1], map2x, map2y, cv.INTER_LINEAR)

# Compute the disparity map using Semi-Global Block Matching (SGBM)
window_size = 5
min_disp = 0
num_disp = 128
stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
)

disparity = stereo.compute(rectified1, rectified2).astype(np.float32) / 16

# Reproject the disparity map to 3D points
points_3D = cv.reprojectImageTo3D(disparity, Q)

# Apply a threshold to remove points that are too close or too far away
min_distance = 300  # minimum distance in millimeters
max_distance = 50000  # maximum distance in millimeters

mask = (
    (disparity > min_disp)
    & (disparity < num_disp)
    & (points_3D[:, :, 2] > min_distance)
    & (points_3D[:, :, 2] < max_distance)
)
filtered_points_3D = points_3D[mask]

# Visualize the disparity map and the 3D points
plt.figure()
plt.imshow(disparity, cmap="gray")
plt.colorbar()
plt.title("Disparity Map")

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    filtered_points_3D[:, 0],
    filtered_points_3D[:, 1],
    filtered_points_3D[:, 2],
    s=0.1,
    c=filtered_points_3D[:, 2],
    cmap="viridis",
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Point Cloud")
plt.show()

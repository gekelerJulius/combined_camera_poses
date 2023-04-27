import math
import os
import cv2 as cv

import numpy as np
from matplotlib import pyplot as plt

from classes.camera_data import CameraData
from classes.logger import Logger
from enums.logging_levels import LoggingLevel
from functions.funcs import (
    extract_extrinsics_from_xml,
    extract_intrinsics_from_xml,
    match_features,
)

img_width = 1920
img_height = 1080

wildtrack_path = "E:\\Uni\\Bachelor\\Wildtrack_dataset"

# Path to the folder containing the images
images_path = os.path.join(wildtrack_path, "Image_subsets")
camera_paths = [os.path.join(images_path, f"C{i}") for i in range(1, 8)]

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

cam_datas = []
dist_coeffs = []
for i in range(1, 8):
    rv, tv = extract_extrinsics_from_xml(extrinsics_paths[i - 1])
    camera_matrix, dist_coeff = extract_intrinsics_from_xml(intrinsics_paths[i - 1])

    cam_datas.append(
        CameraData(
            camera_matrix[0][0],
            camera_matrix[1][1],
            camera_matrix[0][2],
            camera_matrix[1][2],
            1,
            tv,
            rv,
        )
    )
    dist_coeffs.append(dist_coeff)

used_camera_indexes = [0, 5]

Logger.log(
    LoggingLevel.INFO,
    f"Using cameras: {used_camera_indexes} with Extrinsics from {[extrinsics_paths[i] for i in used_camera_indexes]} and Intrinsics from {[intrinsics_paths[i] for i in used_camera_indexes]}",
)

first_image_paths = [
    os.path.join(cam_path, "00000000.png") for cam_path in camera_paths
]

undistorted_images = []
for index in used_camera_indexes:
    image = cv.imread(first_image_paths[index])
    h, w = image.shape[:2]
    cam_data = cam_datas[index]

    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
        cam_data.intrinsic_matrix,
        dist_coeffs[index],
        (w, h),
        1,
        (w, h),
    )

    dst = cv.undistort(
        image, cam_data.intrinsic_matrix, dist_coeffs[index], None, new_camera_matrix
    )

    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]
    undistorted_images.append(dst)

# for i in range(len(undistorted_images)):
#     cv.imshow(f"img{i}", undistorted_images[i])
#     cv.resizeWindow(f"img{i}", img_width, img_height)
#     cv.setMouseCallback(
#         f"img{i}",
#         lambda event, x, y, flags, param: print(f"[{x}, {y}],")
#         if event == cv.EVENT_LBUTTONDOWN
#         else None,
#     )
# cv.waitKey(0)

test_point_array = np.array(
    [
        [
            [1017, 363],
            [953, 353],
            [878, 343],
            [835, 356],
            [783, 372],
            [854, 387],
            [908, 402],
            [970, 380],
        ],
        [
            [565, 302],
            [497, 306],
            [421, 307],
            [436, 317],
            [462, 334],
            [542, 332],
            [616, 326],
            [584, 313],
        ],
    ]
)

H, mask = cv.findHomography(test_point_array[0], test_point_array[1])
inv_H = np.linalg.inv(H)
#
# img_points = np.array(
#     [
#         [1028, 370],
#         [601, 306],
#     ]
# )

# u1, v1 = img_points[0]
# world_point1 = np.matmul(inv_H, np.array([u1, v1, 1]))
# world_point1 = world_point1 / world_point1[2]
#
# print(u1, v1)
# print(world_point1)
#
# u2, v2 = img_points[1]
# world_point2 = np.matmul(inv_H, np.array([u2, v2, 1]))
# world_point2 = world_point2 / world_point2[2]
#
# print(u2, v2)
# print(world_point2)
#
# dist_between_world_points = math.sqrt(
#     (world_point1[0] - world_point2[0]) ** 2
#     + (world_point1[1] - world_point2[1]) ** 2
#     + (world_point1[2] - world_point2[2]) ** 2
# )
# print(dist_between_world_points)
#

src_img = undistorted_images[0]
dst_img = undistorted_images[1]

warped_img = cv.warpPerspective(src_img, inv_H, (dst_img.shape[1], dst_img.shape[0]))

result = cv.addWeighted(dst_img, 0.5, warped_img, 0.5, 0)

cv.imshow("Warped", warped_img)
cv.imshow("result", result)

cv.waitKey(0)

cv.destroyAllWindows()

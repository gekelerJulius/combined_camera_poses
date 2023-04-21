import cv2 as cv
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

images = []

for i in range(0, 49):
    read_image = cv.imread(
        f"pedestrian_zone/data/image_{'0' + str(i) if i < 10 else i}.png",
        cv.IMREAD_GRAYSCALE,
    )
    rotated_image = cv.rotate(read_image, cv.ROTATE_90_COUNTERCLOCKWISE)
    images.append(rotated_image)

# Feature extraction and matching
sift = cv.SIFT_create()
matcher = cv.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": 50})
features = []
matches = []
for img in images:
    kp, des = sift.detectAndCompute(img, None)
    features.append({"kp": kp, "des": des})
for i in range(len(images) - 1):
    matches12 = matcher.knnMatch(features[i]["des"], features[i + 1]["des"], k=2)
    matches21 = matcher.knnMatch(features[i + 1]["des"], features[i]["des"], k=2)
    good_matches12 = []
    good_matches21 = []
    for m, n in matches12:
        if m.distance < 0.6 * n.distance:
            good_matches12.append(m)
    for m, n in matches21:
        if m.distance < 0.6 * n.distance:
            good_matches21.append(m)
    matches.append({"matches12": good_matches12, "matches21": good_matches21})

# Camera calibration
K = np.eye(3)

# Camera pose estimation and 3D point triangulation
points3d = []
for i in range(len(matches)):
    # Camera pose estimation
    E, mask = cv.findEssentialMat(
        np.array([features[i]["kp"][m.queryIdx].pt for m in matches[i]["matches12"]]),
        np.array(
            [features[i + 1]["kp"][m.trainIdx].pt for m in matches[i]["matches12"]]
        ),
        K,
        cv.RANSAC,
        0.7,
        5.0,
        mask=None,
    )
    _, R, t, mask = cv.recoverPose(
        E,
        np.array([features[i]["kp"][m.queryIdx].pt for m in matches[i]["matches12"]]),
        np.array(
            [features[i + 1]["kp"][m.trainIdx].pt for m in matches[i]["matches12"]]
        ),
        K,
        mask=None,
    )
    # 3D point triangulation
    points4d = cv.triangulatePoints(
        np.hstack((np.eye(3), np.zeros((3, 1)))),
        np.hstack((R, t)),
        np.array([features[i]["kp"][m.queryIdx].pt for m in matches[i]["matches12"]]).T,
        np.array(
            [features[i + 1]["kp"][m.trainIdx].pt for m in matches[i]["matches12"]]
        ).T,
    )
    points3d.append(points4d / points4d[3])

points3d = np.concatenate(points3d, axis=1)
points3d = points3d.astype(np.float32)

# Create an Open3D point cloud object
pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points3d[:3, :].T))

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

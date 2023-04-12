import os
from multiprocessing.pool import ThreadPool

import cv2 as cv
import mediapipe as mp
from typing import List
import itertools as it

from common import Timer
from find_obj import init_feature, filter_matches, explore_match

import numpy as np

from classes.person import Person
from functions.get_yolo_boxes import getYoloBoundingBoxes


def affine_skew(tilt, phi, img, mask=None):
    """
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai
    Ai - is an affine transform matrix from skew_img to img
    """
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(
            img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE
        )
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(
            img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv.INTER_NEAREST
        )
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    """
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs
    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.
    ThreadPool object may be passed to speedup the computation.
    """
    params = [(1.0, 0.0)]
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print("affine sampling: %d / %d\r" % (i + 1, len(params)), end="")
        keypoints.extend(k)
        descrs.extend(d)

    print()
    return keypoints, np.array(descrs)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


def annotate_video_multi(file1_name: str, file2_name: str):
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    file1_pre, file1_ext = os.path.splitext(file1_name)
    out1 = cv.VideoWriter(f"{file1_pre}_annotated{file1_ext}", fourcc, 24, (640, 480))
    cap1 = cv.VideoCapture(file1_name)
    file2_pre, file2_ext = os.path.splitext(file2_name)
    out2 = cv.VideoWriter(f"{file2_pre}_annotated{file2_ext}", fourcc, 24, (640, 480))
    cap2 = cv.VideoCapture(file2_name)
    frame_count = 0

    while cap1.isOpened() and cap2.isOpened():
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()
        if not (ret1 and ret2):
            break

        if (
            img1.shape[0] == 0
            or img1.shape[1] == 0
            or img2.shape[0] == 0
            or img2.shape[1] == 0
        ):
            print("Bad frame shape")
            break

        if img1.shape[0] != 480 or img1.shape[1] != 640:
            img1 = cv.resize(img1, (640, 480))
        if img2.shape[0] != 480 or img2.shape[1] != 640:
            img2 = cv.resize(img2, (640, 480))

        persons1: List[Person] = []
        persons2: List[Person] = []
        frame_count += 1
        bounding_boxes1 = getYoloBoundingBoxes(img1)
        bounding_boxes2 = getYoloBoundingBoxes(img2)

        ########################################################################################

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        detector = cv.xfeatures2d.SURF_create()
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)

        matcher = cv.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)

        # pool = ThreadPool(processes=cv.getNumberOfCPUs())
        # kp1, desc1 = affine_detect(detector, img1, pool=pool)
        # kp2, desc2 = affine_detect(detector, img2, pool=pool)
        # print("img1 - %d features, img2 - %d features" % (len(kp1), len(kp2)))
        #
        # with Timer("matching"):
        #     raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
        # p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        # if len(p1) >= 4:
        #     H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
        #     print("%d / %d  inliers/matched" % (np.sum(status), len(status)))
        #     # do not draw outliers (there will be a lot of them)
        #     kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        # else:
        #     H, status = None, None
        #     print("%d matches found, not enough for homography estimation" % len(p1))
        #
        # explore_match("affine find_obj", gray1, gray2, kp_pairs, None, H)
        # cv.waitKey()
        # print("Done")

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        img3 = cv.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good[:10],
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv.imshow("Matches", img3)
        cv.waitKey(0)

        ########################################################################################

        # Get the corresponding points
        # pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        # pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # # First 16 points of each frame
        # if len(kp1) < 8 or len(kp2) < 8:
        #     print("Not enough keypoints")
        #     continue

        # kp1 = kp1[:8]
        # kp2 = kp2[:8]
        #
        # kp1_np = cv.KeyPoint_convert(kp1)
        # kp2_np = cv.KeyPoint_convert(kp2)
        #
        # # Step 2: Normalize points
        # pts1_norm, T1 = normalize_points(kp1_np)
        # pts2_norm, T2 = normalize_points(kp2_np)

        # Step 3: Calculate fundamental matrix
        # F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

        # Select only the inlier points
        # pts1 = pts1[mask.ravel() == 1]
        # pts2 = pts2[mask.ravel() == 1]

        # Compute the epilines for the points in kp1
        # lines1 = cv.computeCorrespondEpilines(pts1, 1, F)
        # lines1 = lines1.reshape(-1, 3)

        #     TODO: 3D reconstruction using epilines:
        #     Compute the fundamental matrix F between the two views, using correspondences between points in the two images.
        #     Compute the epipolar lines corresponding to the points in one of the images using cv.computeCorrespondEpilines(). The input to this function is the set of 2D points in the first image, and the output is a set of corresponding epilines in the second image.
        #     Repeat step 2 for the other image.
        #     For each pair of corresponding epilines, compute the intersection point. This is done by finding the cross product of the two epilines, which gives a 3D line passing through the intersection point. Then, intersect this line with the image planes to obtain the 3D point.
        #     Repeat step 4 for all pairs of corresponding epilines.
        #     Finally, refine the 3D points using bundle adjustment or other optimization techniques to improve the accuracy of the reconstruction.

        #
        # for box in bounding_boxes1:
        #     frame1, landmarks1 = box_to_landmarks_list(frame1, box)
        #     persons1.append(Person(len(persons1), frame_count, box, landmarks1))
        #
        # for box in bounding_boxes2:
        #     frame2, landmarks2 = box_to_landmarks_list(frame2, box)
        #     persons2.append(Person(len(persons2), frame_count, box, landmarks2))
        #
        # for person in persons1:
        #     frame1 = person.draw(frame1)
        #
        # for person in persons2:
        #     frame2 = person.draw(frame2)

        # TODO: Get the landmarks in a shared coordinate system
        #     (i.e. the same coordinate system for both videos) and
        #     then compare the landmarks to see if they are the same person

        out1.write(img1)
        out2.write(img2)

        if frame_count % 10 == 0:
            print(f"Frame {frame_count} done")

        # Debug
        # if frame_count == 50:
        #     break

    cap1.release()
    out1.release()
    cap2.release()
    out2.release()


annotate_video_multi("sim.mp4", "sim2.mp4")

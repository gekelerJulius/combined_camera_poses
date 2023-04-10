import os
import cv2
import mediapipe as mp
from typing import List

import numpy as np

from classes.person import Person
from functions.funcs import (
    get_pose,
    get_landmarks_as_coordinates,
    draw_landmarks_list,
    box_to_landmarks_list,
    are_landmarks_the_same,
    normalize_points,
)
from functions.get_yolo_boxes import getYoloBoundingBoxes

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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    file1_pre, file1_ext = os.path.splitext(file1_name)
    out1 = cv2.VideoWriter(f"{file1_pre}_annotated{file1_ext}", fourcc, 24, (640, 480))
    cap1 = cv2.VideoCapture(file1_name)
    file2_pre, file2_ext = os.path.splitext(file2_name)
    out2 = cv2.VideoWriter(f"{file2_pre}_annotated{file2_ext}", fourcc, 24, (640, 480))
    cap2 = cv2.VideoCapture(file2_name)
    frame_count = 0

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            break

        if (
            frame1.shape[0] == 0
            or frame1.shape[1] == 0
            or frame2.shape[0] == 0
            or frame2.shape[1] == 0
        ):
            print("Bad frame shape")
            break

        if frame1.shape[0] != 480 or frame1.shape[1] != 640:
            frame1 = cv2.resize(frame1, (640, 480))
        if frame2.shape[0] != 480 or frame2.shape[1] != 640:
            frame2 = cv2.resize(frame2, (640, 480))

        persons1: List[Person] = []
        persons2: List[Person] = []
        frame_count += 1
        bounding_boxes1 = getYoloBoundingBoxes(frame1)
        bounding_boxes2 = getYoloBoundingBoxes(frame2)

        try:
            # Step 1: Get points using SIFT
            # sift1 = cv2.SIFT_create()
            # kp1 = sift1.detect(frame1, None)
            # sift2 = cv2.SIFT_create()
            # kp2 = sift2.detect(frame2, None)

            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            # Get the keypoints and descriptors
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)

            # Match the keypoints
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Get the corresponding points
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # # First 16 points of each frame
            # if len(kp1) < 8 or len(kp2) < 8:
            #     print("Not enough keypoints")
            #     continue

            # kp1 = kp1[:8]
            # kp2 = kp2[:8]
            #
            # kp1_np = cv2.KeyPoint_convert(kp1)
            # kp2_np = cv2.KeyPoint_convert(kp2)
            #
            # # Step 2: Normalize points
            # pts1_norm, T1 = normalize_points(kp1_np)
            # pts2_norm, T2 = normalize_points(kp2_np)

            # Step 3: Calculate fundamental matrix
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

            # Select only the inlier points
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

            # Compute the epilines for the points in kp1
            lines1 = cv2.computeCorrespondEpilines(pts1, 1, F)
            lines1 = lines1.reshape(-1, 3)

        #     TODO: 3D reconstruction using epilines:
        #     Compute the fundamental matrix F between the two views, using correspondences between points in the two images.
        #     Compute the epipolar lines corresponding to the points in one of the images using cv2.computeCorrespondEpilines(). The input to this function is the set of 2D points in the first image, and the output is a set of corresponding epilines in the second image.
        #     Repeat step 2 for the other image.
        #     For each pair of corresponding epilines, compute the intersection point. This is done by finding the cross product of the two epilines, which gives a 3D line passing through the intersection point. Then, intersect this line with the image planes to obtain the 3D point.
        #     Repeat step 4 for all pairs of corresponding epilines.
        #     Finally, refine the 3D points using bundle adjustment or other optimization techniques to improve the accuracy of the reconstruction.

        except Exception as e:
            print(e)
            print(F)
            continue

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

        out1.write(frame1)
        out2.write(frame2)

        if frame_count % 10 == 0:
            print(f"Frame {frame_count} done")

        # Debug
        if frame_count == 50:
            break

    cap1.release()
    out1.release()
    cap2.release()
    out2.release()


annotate_video_multi("sim.mp4", "sim2.mp4")

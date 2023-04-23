import os
import time
import cv2.cv2 as cv
import mediapipe as mp
from typing import List, Tuple

import numpy as np

from classes.bounding_box import BoundingBox

from classes.camera_data import CameraData
from classes.custom_point import CustomPoint
from classes.person import Person
from functions.funcs import (
    box_to_landmarks_list,
    draw_landmarks_list,
    compare_landmarks,
)
from functions.get_person_pairs import get_person_pairs
from functions.get_pose import get_pose
from functions.get_yolo_boxes import get_yolo_bounding_boxes

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


def annotate_video_multi(
    file1_name: str,
    file2_name: str,
    cam1_data_path: str = None,
    cam2_data_path: str = None,
):
    if cam1_data_path is None:
        raise ValueError("cam1_data_path must be specified")
    if cam2_data_path is None:
        raise ValueError("cam2_data_path must be specified")
    if file1_name is None:
        raise ValueError("file1_name must be specified")
    if file2_name is None:
        raise ValueError("file2_name must be specified")
    if not os.path.exists(file1_name):
        raise ValueError(f"file1_name does not exist: {file1_name}")
    if not os.path.exists(file2_name):
        raise ValueError(f"file2_name does not exist: {file2_name}")
    if not os.path.exists(cam1_data_path):
        raise ValueError(f"cam1_data_path does not exist: {cam1_data_path}")
    if not os.path.exists(cam2_data_path):
        raise ValueError(f"cam2_data_path does not exist: {cam2_data_path}")

    cam1_data: CameraData = CameraData.create_from_json(cam1_data_path)
    cam2_data: CameraData = CameraData.create_from_json(cam2_data_path)

    cam1_data.test_valid()
    cam2_data.test_valid()
    print("Checks passed, starting video...")

    # fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # file1_pre, file1_ext = os.path.splitext(file1_name)
    # out1 = cv.VideoWriter(f"{file1_pre}_annotated{file1_ext}", fourcc, 24, (640, 480))
    cap1 = cv.VideoCapture(file1_name)
    # file2_pre, file2_ext = os.path.splitext(file2_name)
    # out2 = cv.VideoWriter(f"{file2_pre}_annotated{file2_ext}", fourcc, 24, (640, 480))
    cap2 = cv.VideoCapture(file2_name) if file2_name else cv.VideoCapture(1)
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
        bounding_boxes1: List[BoundingBox] = get_yolo_bounding_boxes(img1)
        bounding_boxes2: List[BoundingBox] = get_yolo_bounding_boxes(img2)

        for box in bounding_boxes1:
            img1, results1 = get_pose(img1, box)
            # img1 = draw_landmarks_list(img1, landmarks1, with_index=True)
            persons1.append(
                Person(f"Person1 {len(persons1)}", frame_count, box, results1)
            )

        for box in bounding_boxes2:
            img2, results2 = get_pose(img2, box)
            # img2 = draw_landmarks_list(img2, landmarks2, with_index=True)
            persons2.append(
                Person(f"Person1 {len(persons2)}", frame_count, box, results2)
            )

        pairs: List[Tuple[Person, Person]] = get_person_pairs(persons1, persons2)

        for p1, p2 in pairs:
            img1 = p1.draw(img1)
            img2 = p2.draw(img2)

        # print(f"Smallest diff: {smallest_diff}")
        # print(f"Rotation: {smallest_diff_rotation}")
        # copy_img2 = img2.copy()
        # copy_img2 = smallest_diff_person.draw(copy_img2)
        # cv.imshow("Frame 2", copy_img2)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # for person in persons1:
        #     img1 = person.draw(img1)
        #
        # for person in persons2:
        #     img2 = person.draw(img2)

        cv.imshow("Frame 1", img1)
        cv.imshow("Frame 2", img2)
        cv.waitKey(0)
        # out1.write(img1)
        # out2.write(img2)

        if frame_count % 10 == 0:
            print(f"Frame {frame_count} done")

        # Debug
        if frame_count == 50:
            break

    cap1.release()
    cap2.release()
    # out1.release()
    # out2.release()


annotate_video_multi(
    "simulation_data/sim1.mp4",
    "simulation_data/sim2.mp4",
    "simulation_data/Cam1_data.json",
    "simulation_data/Cam2_data.json",
)

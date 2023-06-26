import math
import os
import time
from typing import List, Tuple, Optional

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from cv2 import VideoWriter
from numpy import ndarray
from ultralytics import YOLO

from classes.bounding_box import BoundingBox
from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from classes.person_recorder import PersonRecorder
from classes.record_matcher import RecordMatcher
from classes.score_manager import ScoreManager
from classes.true_person_loader import TruePersonLoader
from classes.unity_person import UnityPerson
from enums.logging_levels import LoggingLevel
from functions.funcs import normalize_image, is_in_image
from functions.get_pose import get_pose
from functions.get_yolo_boxes import get_yolo_bounding_boxes

resolution = (1280, 720)


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
    rotation_between_cameras: ndarray = cam1_data.rotation_between_cameras(cam2_data)
    translation_between_cameras: ndarray = cam1_data.translation_between_cameras(
        cam2_data
    )

    # model = YOLO("yolov5nu.pt")
    model = YOLO("yolov8n.pt")
    Logger.log("Starting analysis...", LoggingLevel.INFO)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    file1_pre, file1_ext = os.path.splitext(file1_name)
    file2_pre, file2_ext = os.path.splitext(file2_name)
    cap1 = cv.VideoCapture(file1_name)
    cap2 = cv.VideoCapture(file2_name)
    out1: Optional[VideoWriter] = None
    out2: Optional[VideoWriter] = None
    frame_count = 0
    person_recorder1: PersonRecorder = PersonRecorder("1", 1)
    person_recorder2: PersonRecorder = PersonRecorder("2", 1)
    records_matcher: RecordMatcher = RecordMatcher(person_recorder1, person_recorder2)
    score_manager = ScoreManager()
    start_time = time.time()
    while cap1.isOpened() and cap2.isOpened():
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()

        if not (ret1 and ret2):
            break

        frame_count += 1
        START_FRAME = 55
        END_FRAME = math.inf

        if frame_count < START_FRAME:
            continue
        if frame_count > END_FRAME:
            break

        print(f"Frame {frame_count}")

        if (
            img1.shape[0] == 0
            or img1.shape[1] == 0
            or img2.shape[0] == 0
            or img2.shape[1] == 0
        ):
            Logger.log(
                f"Invalid frame size: {img1.shape}, {img2.shape}",
                LoggingLevel.ERROR,
            )
            break

        # Undistort images
        img1 = cv.undistort(img1, cam1_data.intrinsic_matrix, None)
        img2 = cv.undistort(img2, cam2_data.intrinsic_matrix, None)

        orig_img1 = img1.copy()
        orig_img2 = img2.copy()
        img1 = normalize_image(img1)
        img2 = normalize_image(img2)

        persons1: List[Person] = []
        persons2: List[Person] = []
        bounding_boxes1: List[BoundingBox] = get_yolo_bounding_boxes(orig_img1, model)
        bounding_boxes2: List[BoundingBox] = get_yolo_bounding_boxes(orig_img2, model)

        for box in bounding_boxes1:
            box.draw(img1)
            img1, results1 = get_pose(orig_img1, box)
            if (
                results1 is None
                or results1.pose_landmarks is None
                or results1.pose_world_landmarks is None
            ):
                continue
            length = len([x for x in results1.pose_landmarks.landmark])
            if length > 0:
                persons1.append(
                    Person(frame_count=frame_count, bounding_box=box, results=results1)
                )

        for box in bounding_boxes2:
            box.draw(img2)
            img2, results2 = get_pose(orig_img2, box)
            if (
                results2 is None
                or results2.pose_landmarks is None
                or results2.pose_world_landmarks is None
            ):
                continue
            length = len([x for x in results2.pose_landmarks.landmark])
            if length > 0:
                persons2.append(
                    Person(frame_count=frame_count, bounding_box=box, results=results2)
                )

        person_recorder1.add(persons1, frame_count, img1)
        person_recorder2.add(persons2, frame_count, img2)
        person_recorder1.plot_trajectories(img1)
        person_recorder2.plot_trajectories(img2)

        # Get some records before starting to match
        # if frame_count < START_FRAME + 6:
        #     continue

        # records_matcher.eval_frame(frame_count, img1, img2, cam1_data, cam2_data)
        pairs: List[Tuple[Person, Person]] = []
        # pairs: List[Tuple[Person, Person]] = records_matcher.get_alignment(
        #     frame_count, cam1_data, cam2_data
        # )

        unity_persons: List[UnityPerson] = TruePersonLoader.load(
            "simulation_data/persons"
        )

        # UnityPerson.draw_persons(
        #     unity_persons, [img1, img2], [cam1_data, cam2_data], frame_count
        # )

        pairs = sorted(pairs, key=lambda x: x[0].name)
        for i, (p1, p2) in enumerate(pairs):
            # color1 = p1.color
            # color2 = p2.color
            # blended1, blended2 = blend_colors(color1, color2, 0.5)
            # p1.color = blended1
            # p2.color = blended2

            # color = [0, 0, 0]
            # color[0] = 0 if i % 2 == 0 else 255
            # color[1] = 0 if i % 4 < 2 else 255
            # color[2] = 0 if i < 4 else 255
            # color = tuple(color)

            p2.color = p1.color
            color = p1.color
            p1.draw(img1, color)
            p2.draw(img2, color)
            centroid1 = p1.centroid()
            centroid2 = p2.centroid()
            confirmed = (
                not is_in_image(img1, centroid1[0], centroid1[1])
                or not is_in_image(img2, centroid2[0], centroid2[1])
                or TruePersonLoader.confirm_pair(
                    (p1, p2), unity_persons, frame_count, img1, img2
                )
            )

            score_manager.add_score(1 if confirmed else 0)

        if frame_count > START_FRAME:
            cv.imshow("Frame 1", img1)
            cv.imshow("Frame 2", img2)
            cv.waitKey(100)

        if frame_count == START_FRAME + 1:
            plt.pause(4)

        if out1 is None:
            out1 = cv.VideoWriter(
                f"{file1_pre}_annotated{file1_ext}",
                fourcc,
                24,
                (img1.shape[1], img1.shape[0]),
            )
        if out2 is None:
            out2 = cv.VideoWriter(
                f"{file2_pre}_annotated{file2_ext}",
                fourcc,
                24,
                (img2.shape[1], img2.shape[0]),
            )
        out1.write(img1)
        out2.write(img2)
        score = score_manager.get_score()
        Logger.log(f"Correct percentage: {score * 100:.2f}%", LoggingLevel.INFO)

    end_time = time.time()
    time_taken = end_time - start_time
    frames_per_second = frame_count / time_taken
    Logger.log(f"Frames per second: {frames_per_second:.2f}", LoggingLevel.INFO)
    cap1.release()
    cap2.release()
    out1.release()
    out2.release()

    score = score_manager.get_score()
    records_matcher.report(cam1_data, cam2_data)
    correct_percentage_str = f"Correct percentage: {score * 100:.2f}%"
    # Save correct percentage to file
    with open("simulation_data/correct_percentage.txt", "w") as f:
        f.write(correct_percentage_str)

    Logger.log(correct_percentage_str, LoggingLevel.INFO)
    Logger.log("Done!", LoggingLevel.INFO)
    plt.pause(500000)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    annotate_video_multi(
        "simulation_data/sim1.mp4",
        "simulation_data/sim2.mp4",
        "simulation_data/cam1.json",
        "simulation_data/cam2.json",
    )


if __name__ == "__main__":
    main()

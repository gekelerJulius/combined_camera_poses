import math
import os
import time
from typing import List, Tuple

import cv2 as cv
from ultralytics import YOLO

from classes.bounding_box import BoundingBox
from classes.camera_data import CameraData
from classes.logger import Logger, Divider
from classes.person import Person
from classes.person_recorder import PersonRecorder
from classes.record_matcher import RecordMatcher
from classes.score_manager import ScoreManager
from classes.true_person_loader import TruePersonLoader
from classes.unity_person import UnityPerson
from enums.logging_levels import LoggingLevel
from functions.funcs import blend_colors, normalize_image
from functions.get_person_pairs import match_pairs
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

    R_between_cameras = cam1_data.rotation_between_cameras(cam2_data)
    t_between_cameras = cam1_data.translation_between_cameras(cam2_data)
    with Divider("Camera Data"):
        Logger.log(R_between_cameras, LoggingLevel.INFO)
        Logger.log(t_between_cameras, LoggingLevel.INFO)

    model = YOLO()
    Logger.log("Starting analysis...", LoggingLevel.INFO)

    # test_joint_names = ['l_wrist', 'l_elbow']
    # Joint index pairs specifying which ones should be connected with a line (i.e., the bones of
    # the body, e.g. wrist-elbow, elbow-shoulder)
    # test_joint_edges = [[0, 1]]
    # viz = poseviz.PoseViz(NAMES_LIST, CONNECTIONS_LIST)
    # viz = poseviz.PoseViz(test_joint_names, test_joint_edges)

    # fourcc = cv.VideoWriter_fourcc(*"mp4v")
    # file1_pre, file1_ext = os.path.splitext(file1_name)
    # file2_pre, file2_ext = os.path.splitext(file2_name)
    cap1 = cv.VideoCapture(file1_name)
    cap2 = cv.VideoCapture(file2_name)
    # out1: VideoWriter = None
    # out2: VideoWriter = None
    frame_count = 0
    person_recorder1: PersonRecorder = PersonRecorder("1")
    person_recorder2: PersonRecorder = PersonRecorder("2")
    records_matcher: RecordMatcher = RecordMatcher(person_recorder1, person_recorder2)
    score_manager = ScoreManager()
    img1 = None
    img2 = None
    pairs: List[Tuple[Person, Person]] = []
    while cap1.isOpened() and cap2.isOpened():
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()
        if not (ret1 and ret2):
            break

        frame_count += 1
        START_FRAME = 0
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
                f"Invalid frame size: {img1.shape}, {img2.shape}", LoggingLevel.ERROR
            )
            break

        # Undistort images
        img1 = cv.undistort(img1, cam1_data.intrinsic_matrix, None)
        img2 = cv.undistort(img2, cam2_data.intrinsic_matrix, None)

        orig_img1 = img1.copy()
        orig_img2 = img2.copy()
        img1 = normalize_image(img1)
        img2 = normalize_image(img2)

        # Save img1 to disk
        # cv.imwrite(f"img1_{frame_count}.jpg", img1)

        persons1: List[Person] = []
        persons2: List[Person] = []
        bounding_boxes1: List[BoundingBox] = get_yolo_bounding_boxes(orig_img1, model)
        bounding_boxes2: List[BoundingBox] = get_yolo_bounding_boxes(orig_img2, model)

        for box in bounding_boxes1:
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
                    Person(f"Person1 {len(persons1)}", frame_count, box, results1)
                )

        for box in bounding_boxes2:
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
                    Person(f"Person2 {len(persons2)}", frame_count, box, results2)
                )

        person_recorder1.add(persons1, frame_count, img1)
        person_recorder2.add(persons2, frame_count, img2)
        records_matcher.eval_frame(frame_count, img1, img2, cam1_data, cam2_data)

        if frame_count < 15:
            continue

        # for person in unity_persons:
        #     print(person)
        #     pts1: ndarray = person.get_image_points(frame_count, cam1_data)
        #     pts2: ndarray = person.get_image_points(frame_count, cam2_data)
        #
        #     assert pts1 is not None and pts2 is not None
        #     assert len(pts1) == len(pts2) != 0
        #
        #     for i in range(len(pts1)):
        #         pt1 = tuple(pts1[i].astype(int))
        #         pt2 = tuple(pts2[i].astype(int))
        #         cv.circle(img1, pt1, 5, (255, 255, 0), -1)
        #         cv.circle(img2, pt2, 5, (255, 255, 0), -1)

        # if frame_count < 15:
        #     continue

        # pairs = match_pairs(
        #     person_recorder1,
        #     person_recorder2,
        #     frame_count,
        #     img1,
        #     img2,
        #     cam1_data,
        #     cam2_data,
        # )

        pairs = records_matcher.get_alignment(frame_count)

        unity_persons: List[UnityPerson] = TruePersonLoader.load(
            "simulation_data/persons"
        )

        for i, (p1, p2) in enumerate(pairs):
            color1 = p1.color
            color2 = p2.color
            blended1, blended2 = blend_colors(color1, color2, 0)
            p1.color = blended1
            p2.color = blended2
            color = [0, 0, 0]
            color[i] = 255
            color = tuple(color)
            p1.draw(img1, color)
            p2.draw(img2, color)
            confirmed = TruePersonLoader.confirm_pair(
                (p1, p2), unity_persons, frame_count, cam1_data, cam2_data
            )
            score_manager.add_score(1 if confirmed else 0)

        if frame_count > START_FRAME:
            cv.imshow("Frame 1", img1)
            cv.imshow("Frame 2", img2)
            cv.waitKey(1)

        # if out1 is None:
        #     out1 = cv.VideoWriter(f"{file1_pre}_annotated{file1_ext}", fourcc, 24, (img1.shape[1], img1.shape[0]))
        # if out2 is None:
        #     out2 = cv.VideoWriter(f"{file2_pre}_annotated{file2_ext}", fourcc, 24, (img2.shape[1], img2.shape[0]))

        # out1.write(img1)
        # out2.write(img2)
        if frame_count % 10 == 0:
            Logger.log(f"Frame: {frame_count}", LoggingLevel.INFO)
            score = score_manager.get_score()
            # Show percentage with 2 decimal places
            Logger.log(f"Correct percentage: {score * 100:.2f}%", LoggingLevel.INFO)

        # Debug
        # if frame_count == 50:
        #     break

    cap1.release()
    cap2.release()
    # out1.release()
    # out2.release()
    score = score_manager.get_score()
    Logger.log(f"Correct percentage: {score * 100:.2f}%", LoggingLevel.INFO)
    Logger.log("Done!", LoggingLevel.INFO)
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

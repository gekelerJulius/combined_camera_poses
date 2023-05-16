import os

from cv2 import VideoWriter
from ultralytics import YOLO

import cv2 as cv
from typing import List, Tuple

from classes.bounding_box import BoundingBox

from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from classes.person_recorder import PersonRecorder
from enums.logging_levels import LoggingLevel
from functions.get_person_pairs import get_person_pairs, get_person_pairs_simple_distance, test_pairing
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
    cam1_data.test_valid(resolution)
    cam2_data.test_valid(resolution)
    model = YOLO()
    Logger.log("Starting analysis...", LoggingLevel.INFO)

    # test_joint_names = ['l_wrist', 'l_elbow']

    # Joint index pairs specifying which ones should be connected with a line (i.e., the bones of
    # the body, e.g. wrist-elbow, elbow-shoulder)
    # test_joint_edges = [[0, 1]]
    # viz = poseviz.PoseViz(NAMES_LIST, CONNECTIONS_LIST)
    # viz = poseviz.PoseViz(test_joint_names, test_joint_edges)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    file1_pre, file1_ext = os.path.splitext(file1_name)
    file2_pre, file2_ext = os.path.splitext(file2_name)
    cap1 = cv.VideoCapture(file1_name)
    cap2 = cv.VideoCapture(file2_name)
    # out1: VideoWriter = None
    # out2: VideoWriter = None
    frame_count = 0
    person_recorder1: PersonRecorder = PersonRecorder()
    person_recorder2: PersonRecorder = PersonRecorder()

    while cap1.isOpened() and cap2.isOpened():
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()
        if not (ret1 and ret2):
            break

        frame_count += 1

        if frame_count < 60:
            continue

        if (
                img1.shape[0] == 0
                or img1.shape[1] == 0
                or img2.shape[0] == 0
                or img2.shape[1] == 0
        ):
            Logger.log(f"Invalid frame size: {img1.shape}, {img2.shape}", LoggingLevel.ERROR)
            break

        # Undistort images
        img1 = cv.undistort(img1, cam1_data.intrinsic_matrix, None)
        img2 = cv.undistort(img2, cam2_data.intrinsic_matrix, None)

        # Save img1 to disk
        # cv.imwrite(f"img1_{frame_count}.jpg", img1)

        persons1: List[Person] = []
        persons2: List[Person] = []
        bounding_boxes1: List[BoundingBox] = get_yolo_bounding_boxes(img1, model)
        bounding_boxes2: List[BoundingBox] = get_yolo_bounding_boxes(img2, model)

        for box in bounding_boxes1:
            img1, results1 = get_pose(img1, box)

            if results1 is None or results1.pose_landmarks is None or results1.pose_world_landmarks is None:
                continue
            length = len([x for x in results1.pose_landmarks.landmark])
            if length > 0:
                persons1.append(
                    Person(f"Person1 {len(persons1)}", frame_count, box, results1)
                )

        for box in bounding_boxes2:
            img2, results2 = get_pose(img2, box)

            if results2 is None or results2.pose_landmarks is None or results2.pose_world_landmarks is None:
                continue
            length = len([x for x in results2.pose_landmarks.landmark])
            if length > 0:
                persons2.append(
                    Person(f"Person2 {len(persons2)}", frame_count, box, results2)
                )

        for person1 in persons1:
            person_recorder1.add(person1)
        person_recorder1.eval(frame_count)

        for person2 in persons2:
            person_recorder2.add(person2)
        person_recorder2.eval(frame_count)

        # TODO: Use past recordings to determine if the pairings are plausible

        pairs: List[Tuple[Person, Person]] = get_person_pairs_simple_distance(
            persons1, persons2, img1, img2, cam1_data, cam2_data
        )

        def get_color(index):
            if index == 0:
                return 0, 0, 255
            elif index == 1:
                return 0, 255, 0
            elif index == 2:
                return 255, 0, 0
            else:
                raise ValueError(f"Invalid index: {index}")

        for i, (p1, p2) in enumerate(pairs):
            color = get_color(i)
            p1.draw(img1, color)
            p2.draw(img2, color)

        # TODO: Add and fix Homography
        # if persons1 and len(persons1) > 0:
        #     p1 = persons1[0]
        #     feet_points_on_img1 = p1.get_feet_points()
        #     feet_points_real1 = p1.get_feet_points_real(cam1_data)
        #     H, mask = cv.findHomography(feet_points_on_img1, feet_points_real1, cv.RANSAC, 5.0)
        #
        #     mean_point = np.mean(p1.get_pose_landmarks_numpy(), axis=0)
        #     mean_point_real = np.matmul(H, mean_point)
        #     ax1.plot(mean_point_real[0], mean_point_real[1], 'ro')
        #     plt.pause(0.01)

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

        # Debug
        # if frame_count == 50:
        #     break

    cap1.release()
    cap2.release()
    # out1.release()
    # out2.release()
    Logger.log("Done!", LoggingLevel.INFO)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    annotate_video_multi(
        "simulation_data/sim1.mp4",
        "simulation_data/sim2.mp4",
        "simulation_data/cam1.json",
        "simulation_data/cam2.json",
    )


if __name__ == '__main__':
    main()

from typing import Generator, Tuple, List

import cv2
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes, Results

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
from functions.get_pose import get_pose

START_FRAME = 0


def main():
    model: YOLO = YOLO("yolov8n")
    vid_paths: List[str] = [
        "simulation_data/sim1.mp4",
        "simulation_data/sim2.mp4",
    ]
    cam_data_paths: List[str] = [
        "simulation_data/cam1.json",
        "simulation_data/cam2.json",
    ]
    generators: List[Generator[Results]] = [
        model.track(vid_path, show=False, conf=0.2, classes=0, stream=True)
        for vid_path in vid_paths
    ]
    zipped: Generator[Tuple[Results, Results]] = zip(*generators)
    person_lists: List[List[Person]] = [[], []]

    # TODO: Tracking Persons is now handled better by yolov8,
    #  so we can remove the tracking part from the person_recorder and use it only to record
    #  past positions of the person.

    person_recorders: List[PersonRecorder] = [
        PersonRecorder("1", 1),
        PersonRecorder("2", 2),
    ]
    records_matcher: RecordMatcher = RecordMatcher(person_recorders)
    score_manager = ScoreManager()
    pairs: List[Tuple[Person, Person]] = []
    frame_count = START_FRAME

    print("Starting...")
    for yolo_results in zipped:
        frame_count += 1

        for i, result in enumerate(yolo_results):
            person_list = person_lists[i]
            boxes: Boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0]
                min_x = int(xyxy[0])
                min_y = int(xyxy[1])
                max_x = int(xyxy[2])
                max_y = int(xyxy[3])
                tracker_id: int = int(box.id)
                # cv2.rectangle(
                #     result.orig_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2
                # )
                # cv2.putText(
                #     result.orig_img,
                #     str(tracker_id),
                #     (min_x, min_y - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.9,
                #     (36, 255, 12),
                #     2,
                # )

                pose_result1 = get_pose(
                    result.orig_img, BoundingBox(min_x, min_y, max_x, max_y)
                )
                if (
                        pose_result1 is None
                        or pose_result1.pose_landmarks is None
                        or pose_result1.pose_landmarks.landmark is None
                ):
                    continue
                length = len([x for x in pose_result1.pose_landmarks.landmark])
                if length > 0:
                    person_list.append(
                        Person(
                            f"Person{i} {len(person_list)}",
                            frame_count,
                            box,
                            pose_result1,
                        )
                    )
            person_recorders[i].add(person_list, frame_count, result.orig_img)

        # Get some records before starting to match
        if frame_count < START_FRAME + 6:
            continue

        img1 = yolo_results[0].orig_img
        img2 = yolo_results[1].orig_img

        cam_datas: List[CameraData] = [
            CameraData.create_from_json(cam_data_path)
            for cam_data_path in cam_data_paths
        ]
        records_matcher.eval_frame(
            frame_count,
            img1,
            img2,
            cam_datas[0],
            cam_datas[1],
        )
        pairs = records_matcher.get_alignment(frame_count, cam_datas[0], cam_datas[1])
        pairs = sorted(pairs, key=lambda x: x[0].name)

        unity_persons: List[UnityPerson] = TruePersonLoader.load(
            "simulation_data/persons"
        )

        for i, (p1, p2) in enumerate(pairs):
            p2.color = p1.color
            color = p1.color
            p1.draw(img1, color)
            p2.draw(img2, color)
            confirmed = TruePersonLoader.confirm_pair(
                (p1, p2), unity_persons, frame_count, img1, img2
            )
            score_manager.add_score(1 if confirmed else 0)

        score = score_manager.get_score()
        Logger.log(f"Correct percentage: {score * 100:.2f}%", LoggingLevel.INFO)
        cv2.imshow("Frame 1", img1)
        cv2.imshow("Frame 2", img2)
        cv2.waitKey(100)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

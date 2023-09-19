from typing import Generator, Tuple, List

import cv2
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes, Results

from classes.alignment_matcher import AlignmentMatcher
from classes.bounding_box import BoundingBox
from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from classes.person_history import PersonHistory
from classes.score_manager import ScoreManager
from classes.true_person_loader import TruePersonLoader
from classes.unity_person import UnityPerson
from enums.logging_levels import LoggingLevel
from functions.get_pose import get_pose

START_FRAME = 0


def main():
    vid_paths: List[str] = [
        "simulation_data/sim1.mp4",
        "simulation_data/sim2.mp4",
    ]
    cam_data_paths: List[str] = [
        "simulation_data/cam1.json",
        "simulation_data/cam2.json",
    ]
    models: List[YOLO] = [YOLO("yolov8n") for _ in enumerate(vid_paths)]

    generators: List[Generator[Results]] = [
        models[i].track(vid_path, show=False, conf=0.2, classes=0, stream=True)
        for i, vid_path in enumerate(vid_paths)
    ]
    zipped: Generator[Tuple[Results, Results]] = zip(*generators)
    person_histories: List[PersonHistory] = [
        PersonHistory() for _ in enumerate(cam_data_paths)
    ]
    alignment_matcher = AlignmentMatcher(person_histories)
    score_manager = ScoreManager()
    pairs: List[Tuple[Person, Person]] = []
    frame_count = 0

    print("Starting...")
    for yolo_results in zipped:
        frame_count += 1
        if frame_count < START_FRAME:
            continue

        for i, result in enumerate(yolo_results):
            i: int = int(i)
            result: Results = result
            persons = []
            boxes: Boxes = result.boxes
            for _ in boxes:
                xyxy = _.xyxy[0]
                min_x = max(0, int(xyxy[0]))
                min_y = max(0, int(xyxy[1]))
                max_x = min(result.orig_img.shape[1], int(xyxy[2]))
                max_y = min(result.orig_img.shape[0], int(xyxy[3]))

                # cv2.rectangle(
                #     result.orig_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2
                # )
                # cv2.putText(
                #     result.orig_img,
                #     str(int(box.id)),
                #     (min_x, min_y - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.9,
                #     (36, 255, 12),
                #     2,
                # )
                bbox = BoundingBox(min_x, min_y, max_x, max_y)
                pose_result = get_pose(result.orig_img, bbox)
                if (
                    pose_result is None
                    or pose_result.pose_landmarks is None
                    or pose_result.pose_landmarks.landmark is None
                ):
                    continue
                length = len([x for x in pose_result.pose_landmarks.landmark])
                if length > 0:
                    person = Person(
                        f"Image {i + 1}, Box {int(_.id[0])}",
                        frame_count,
                        bbox,
                        pose_result,
                    )
                    persons.append(person)

                    p_id = _.id
                    # color depending on id (1 = red, 2 = blue, 3 = green)
                    color = (0, 0, 0)
                    if p_id == 1:
                        color = (0, 0, 255)
                    elif p_id == 2:
                        color = (255, 0, 0)
                    elif p_id == 3:
                        color = (0, 255, 0)
                    person.color = color

            person_histories[i].add(persons, frame_count)

        img1 = yolo_results[0].orig_img
        img2 = yolo_results[1].orig_img

        cam_datas: List[CameraData] = [
            CameraData.create_from_json(cam_data_path)
            for cam_data_path in cam_data_paths
        ]

        # TODO: Think about how to get pairs in the future

        pairs = alignment_matcher.update_alignments(frame_count)
        # pairs = records_matcher.get_alignment(frame_count, cam_datas[0], cam_datas[1])

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
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

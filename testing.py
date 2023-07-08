from typing import Generator

import cv2
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Boxes, Results


def main():
    vid_path: str = "simulation_data/sim2.mp4"
    model: YOLO = YOLO("yolov8n")
    results: Generator[Results] = model.track(vid_path, show=False, conf=0.2, classes=0, stream=True)

    for result in results:
        boxes: Boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0]
            min_x = int(xyxy[0])
            min_y = int(xyxy[1])
            max_x = int(xyxy[2])
            max_y = int(xyxy[3])
            tracker_id: int = int(box.id)
            cv2.rectangle(result.orig_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            cv2.putText(result.orig_img, str(tracker_id), (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (36, 255, 12), 2)
        cv2.imshow("frame", result.orig_img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

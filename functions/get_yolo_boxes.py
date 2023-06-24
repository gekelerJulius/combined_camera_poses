from typing import List

from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results, Boxes

from classes.bounding_box import BoundingBox


def get_yolo_bounding_boxes(image, model: YOLO):
    results: Results = model.predict(
        image, stream=False, conf=0.4, device="cpu", show=False, classes=0
    )[0]
    person_indexes = []
    boxes: Boxes = results.boxes
    for i, cls in enumerate(boxes.cls):
        if cls == 0:
            person_indexes.append(i)

    bb_np = [boxes.xyxy[i] for i in person_indexes]
    bounding_boxes: List[BoundingBox] = []
    for i, box in enumerate(bb_np):
        min_x = int(box[0])
        min_y = int(box[1])
        max_x = int(box[2])
        max_y = int(box[3])
        bounding_boxes.append(BoundingBox(min_x, min_y, max_x, max_y))
    return bounding_boxes

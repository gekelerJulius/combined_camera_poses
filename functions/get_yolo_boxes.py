from typing import List
from yolov5 import YOLOv5
from yolov5.models.common import Detections

from classes.bounding_box import BoundingBox


def get_yolo_bounding_boxes(image, model: YOLOv5):
    results: Detections = model.predict(image)

    # Filter for persons
    predictions = results.pred[0]
    categories = predictions[:, 5]
    label_mask = [results.names[int(c)] for c in categories]
    person_indexes = [i for i, x in enumerate(label_mask) if x == "person"]
    bb_np = results.xyxy[0].numpy()
    bounding_boxes: List[BoundingBox] = []
    for i in range(len(person_indexes)):
        box = bb_np[person_indexes[i]]
        min_x = int(box[0])
        min_y = int(box[1])
        max_x = int(box[2])
        max_y = int(box[3])
        bounding_boxes.append(BoundingBox(min_x, min_y, max_x, max_y))
    return bounding_boxes

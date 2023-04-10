from typing import List

from yolov5 import YOLOv5

from classes.bounding_box import BoundingBox

model_path = "yolov5s.pt"
device = "cpu"
model = YOLOv5(model_path, device)


def getYoloBoundingBoxes(image):
    boxes = model.predict(image)
    bb_np = boxes.xyxy[0].numpy()
    bounding_boxes: List[BoundingBox] = []
    for box in bb_np:
        min_x = int(box[0])
        min_y = int(box[1])
        max_x = int(box[2])
        max_y = int(box[3])
        bounding_boxes.append(BoundingBox(min_x, min_y, max_x, max_y))
    return bounding_boxes

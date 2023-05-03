from mmdet.apis import init_detector, inference_detector
from mmdet.evaluation import coco_classes
from mmdet.structures import DetDataSample
import cv2 as cv
from mmdet.visualization import DetLocalVisualizer

from classes.timer import Timer

config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
visualizer = DetLocalVisualizer()

img = cv.imread('demo/test.jpg')

with Timer('Inference'):
    res: DetDataSample = inference_detector(model, img)

# visualizer.add_datasample('test', img, res, show=True)

class_names = coco_classes()

bboxes = res.pred_instances.get('bboxes')
labels = res.pred_instances.get('labels')
scores = res.pred_instances.get('scores')

for i, bbox in enumerate(bboxes):
    if labels[i] != 0 or scores[i] < 0.5:
        continue
    min_x, min_y, max_x, max_y = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 255, 0), 1)
    cv.putText(img, f"{class_names[labels[i]]} {scores[i]}", (min_x, min_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
               1)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

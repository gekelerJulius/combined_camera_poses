from typing import NamedTuple

import cv2
import numpy as np
from mediapipe.python.solutions.pose import Pose
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

pose = Pose(
    static_image_mode=True,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
img = cv2.imread("test1.jpg")
img = cv2.resize(img, (720, 1080), interpolation=cv2.INTER_AREA)

results: NamedTuple = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
connections = POSE_CONNECTIONS

pts = [
    [lmk.x * img.shape[1], lmk.y * img.shape[0]]
    for lmk in results.pose_landmarks.landmark
]
pts = np.array(pts).astype(np.int32)
for connection in connections:
    cv2.line(
        img,
        (pts[connection[0]][0], pts[connection[0]][1]),
        (pts[connection[1]][0], pts[connection[1]][1]),
        (0, 0, 0),
        2,
    )

for pt in pts:
    cv2.circle(img, (pt[0], pt[1]), 2, (0, 0, 255), -1)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("test1_out.jpg", img)

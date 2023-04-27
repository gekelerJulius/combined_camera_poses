from typing import Tuple

import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark import Landmark


class ColoredLandmark:
    avg_color: Tuple[int, int, int]
    dom_color: Tuple[int, int, int]
    landmark: Landmark
    x: float
    y: float
    z: float
    visibility: float

    def __init__(
        self,
        landmark: Landmark,
        avg_color: Tuple[int, int, int],
        dom_color: Tuple[int, int, int],
    ):
        self.landmark = landmark
        self.avg_color = avg_color
        self.dom_color = dom_color
        self.x = landmark.x
        self.y = landmark.y
        self.z = landmark.z
        self.visibility = landmark.visibility

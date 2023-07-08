from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray


class BoundingBox:
    def __init__(self, min_x: int, min_y: int, max_x: int, max_y: int):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def get_width(self):
        return self.max_x - self.min_x

    def get_height(self):
        return self.max_y - self.min_y

    def __str__(self):
        return f"Min X: {self.min_x} | Min Y: {self.min_y} | Max X: {self.max_x} | Max Y: {self.max_y}"

    def __repr__(self):
        return self.__str__()

    def increase_size(self, increase_factor: float) -> None:
        width_increase = int(self.get_width() * increase_factor)
        height_increase = int(self.get_height() * increase_factor)
        self.min_x -= width_increase
        self.min_y -= height_increase
        self.max_x += width_increase
        self.max_y += height_increase

    def draw(self, img) -> ndarray:
        start_x = int(self.min_x)
        start_y = int(self.min_y)
        end_x = int(self.max_x)
        end_y = int(self.max_y)
        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), thickness=2)
        return img

    def get_center(self):
        return (self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2

    def to_numpy(self):
        return np.array([self.min_x, self.min_y, self.max_x, self.max_y])

    def get_area(self):
        return self.get_width() * self.get_height()

    def as_tracking_window(self) -> Tuple[int, int, int, int]:
        return self.min_x, self.min_y, self.get_width(), self.get_height()

    def crop_image(self, img: ndarray) -> ndarray:
        return img.copy()[self.min_y:self.max_y, self.min_x:self.max_x]

    def calculate_overlap_percentage(self, other: "BoundingBox") -> float:
        if self.min_x > other.max_x or self.max_x < other.min_x:
            return 0
        if self.min_y > other.max_y or self.max_y < other.min_y:
            return 0

        intersection = max(
            0, min(self.max_x, other.max_x) - max(self.min_x, other.min_x)
        ) * max(0, min(self.max_y, other.max_y) - max(self.min_y, other.min_y))

        area = self.get_width() * self.get_height()
        return intersection / area

    @staticmethod
    def from_points(org_points: ndarray) -> "BoundingBox":
        min_x = np.min(org_points[:, 0])
        min_y = np.min(org_points[:, 1])
        max_x = np.max(org_points[:, 0])
        max_y = np.max(org_points[:, 1])
        return BoundingBox(min_x, min_y, max_x, max_y)

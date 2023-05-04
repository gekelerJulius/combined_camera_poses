import cv2
import numpy as np


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

    def draw(self, img):
        cv2.rectangle(
            img, (self.min_x, self.min_y), (self.max_x, self.max_y), (0, 255, 0), 2
        )

    def get_center(self):
        return (self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2

    def to_numpy(self):
        return np.array([self.min_x, self.min_y, self.max_x, self.max_y])

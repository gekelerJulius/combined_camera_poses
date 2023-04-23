import math

import numpy as np


class CustomPoint:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def from_list(point_list):
        return CustomPoint(point_list[0], point_list[1], point_list[2])

    @staticmethod
    def from_tuple(point_tuple):
        return CustomPoint(point_tuple[0], point_tuple[1], point_tuple[2])

    @staticmethod
    def from_np_array(point_array):
        return CustomPoint(point_array[0], point_array[1], point_array[2])

    def as_list(self):
        return [self.x, self.y, self.z]

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self):
        return f"CustomPoint({self.x}, {self.y}, {self.z})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index):
        if index == 0:
            return self.x
        if index == 1:
            return self.y
        if index == 2:
            return self.z
        raise IndexError("Index out of range")

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __add__(self, other):
        return CustomPoint(self.x + other.x, self.y + other.y + self.z + other.z)

    def __sub__(self, other):
        return CustomPoint(self.x - other.x, self.y - other.y + self.z - other.z)

    def __mul__(self, other):
        return CustomPoint(self.x * other.x, self.y * other.y + self.z * other.z)

    def __truediv__(self, other):
        return CustomPoint(self.x / other.x, self.y / other.y + self.z / other.z)

    def distance(self, other):
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def to_np_array(self):
        return np.array([self.x, self.y, self.z])

import math


class CustomPoint:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def as_list(self):
        return [self.x, self.y, self.z]

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self):
        return f"CustomPoint({self.x}, {self.y}, {self.z})"

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

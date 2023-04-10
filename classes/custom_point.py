import math


class CustomPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"CustomPoint({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __add__(self, other):
        return CustomPoint(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return CustomPoint(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return CustomPoint(self.x * other.x, self.y * other.y)

    def __truediv__(self, other):
        return CustomPoint(self.x / other.x, self.y / other.y)

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

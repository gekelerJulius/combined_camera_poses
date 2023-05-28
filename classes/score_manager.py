import numpy as np
from numpy import ndarray


class ScoreManager:
    scores: ndarray

    def __init__(self):
        self.scores = np.array([])

    def add_score(self, score: float):
        self.scores = np.append(self.scores, score)

    def get_score(self):
        if len(self.scores) == 0:
            return 0
        return np.mean(self.scores)

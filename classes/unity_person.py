from typing import List
import json

import numpy as np


class UnityPerson:
    jsonpath: str
    data: dict = {}
    keys: List[str] = []

    def __init__(self, jsonpath: str):
        self.jsonpath = jsonpath
        with open(jsonpath) as f:
            data = json.load(f)
            for a in data:
                self.data[a] = data[a]
            self.keys = list(data.keys())

    def numpy(self) -> np.ndarray:
        x = [val['x'] for val in self.data.values()]
        y = [val['y'] for val in self.data.values()]
        z = [val['z'] for val in self.data.values()]
        return np.array([x, y, z])

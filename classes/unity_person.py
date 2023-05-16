from typing import List
import json

import numpy as np
from numpy import ndarray

from consts.consts import NAMES_LIST
from consts.mixamo_mapping import from_mixamo


class UnityPerson:
    jsonpath: str

    def __init__(self, jsonpath: str):
        self.jsonpath = jsonpath

    def get_frame(self, n: int) -> ndarray:
        org_points = load_points(self.jsonpath + f"/{n}.json")
        return org_points


def load_points(jsonpath: str) -> ndarray:
    org_dict = {}
    with open(jsonpath) as f:
        org_points_json = json.load(f)
    strings_to_remove = ["mixamorig:", "Ch42_", "Walking_Man", "Walking_Woman"]
    old_names = [a for a in org_points_json]
    for i in range(len(org_points_json)):
        name = old_names[i]
        new_name = str(name)
        for s in strings_to_remove:
            new_name = new_name.replace(s, "")

        if len(new_name) == 0:
            continue
        org_points_json[new_name] = org_points_json[name]

    for name in old_names:
        del org_points_json[name]

    for name in [a for a in org_points_json]:
        if name not in from_mixamo:
            del org_points_json[name]
        else:
            mapped: List[str] = from_mixamo[name]
            for m in mapped:
                org_dict[m] = org_points_json[name]

    org_points = []
    for name in NAMES_LIST:
        if name in org_dict:
            val = org_dict[name]
            org_points.append([val["x"], val["y"], val["z"]])

    return np.array(org_points)

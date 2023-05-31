from typing import List
import json

import numpy as np
from numpy import ndarray

from classes.bounding_box import BoundingBox
from classes.camera_data import CameraData
from classes.person import Person
from consts.consts import NAMES_LIST
from consts.mixamo_mapping import from_mixamo
from functions.funcs import plot_pose_3d


class UnityPerson:
    jsonpath: str

    def __init__(self, jsonpath: str):
        self.jsonpath = jsonpath

    def get_frame(self, n: int) -> ndarray:
        org_points = load_points(self.jsonpath + f"/{n}.json")
        return org_points

    def get_image_points(self, frame_count: int, camera_info: CameraData) -> ndarray:
        org_points = self.get_frame(frame_count)
        return camera_info.transform_points_to_camera(org_points)

    __str__ = lambda self: self.jsonpath
    __repr__ = lambda self: self.jsonpath

    @staticmethod
    def plot_all_3d(persons: List["UnityPerson"]) -> None:
        import matplotlib.pyplot as plt

        plot_id = 0
        for person in persons:
            for i in range(0, 100):
                pts = person.get_frame(i)
                # Invert y axis
                pts[:, 1] *= -1
                plot_pose_3d(pts, plot_id)
        plt.show()


def load_points(jsonpath: str) -> ndarray:
    org_dict = {}
    with open(jsonpath) as f:
        org_points_json = json.load(f)
    strings_to_remove = [
        "mixamorig:",
        "Ch42_",
        "Walking_Man",
        "Ch41_",
        "mixamorig4:",
        "Walking_Woman",
    ]
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

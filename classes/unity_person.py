from typing import List, Any
import json

import cv2
import numpy as np
from numpy import ndarray

from classes.bounding_box import BoundingBox
from consts.consts import NAMES_LIST
from consts.mixamo_mapping import from_mixamo

json_cam_keys = ["cam1", "cam2"]


class UnityPerson:
    jsonpath: str

    def __init__(self, jsonpath: str):
        self.jsonpath = jsonpath

    def get_frame(self, n: int, camera_num: int) -> ndarray:
        org_points = load_points(
            self.jsonpath + f"/{n}.json", json_cam_keys[camera_num]
        )
        cv_points = np.copy(org_points)
        return cv_points

    def get_image_points(
        self, frame_count: int, cam_num: int, img_height: int
    ) -> ndarray:
        pts = self.get_frame(frame_count, cam_num)
        pts[:, 1] = img_height - pts[:, 1]
        return pts

    __str__ = lambda self: self.jsonpath
    __repr__ = lambda self: self.jsonpath

    @staticmethod
    def draw_persons(
        persons: List["UnityPerson"],
        images: List[Any],
        frame_count: int,
    ) -> None:
        for i in range(len(images)):
            img = images[i]
            height = img.shape[0]
            for person in persons:
                points = person.get_image_points(frame_count, i, height)
                for point in points:
                    point = np.copy(point).astype(int)
                    cv2.circle(img, tuple(point), 3, (255, 255, 0), -1)

    def get_bounding_box(
        self, frame_count: int, cam_num: int, img_height: int
    ) -> BoundingBox:
        points = self.get_image_points(frame_count, cam_num, img_height)
        return BoundingBox.from_points(points)


def load_points(jsonpath: str, cam_key: str) -> ndarray:
    org_dict = {}
    with open(jsonpath) as f:
        json_data = json.load(f)

    org_points_json = json_data[cam_key]
    strings_to_remove = [
        "mixamorig:",
        "Ch42_",
        "Walking_Man",
        "Ch41_",
        "mixamorig4:",
        "Walking_Woman",
        "Ch28_",
        "mixamorig10:",
        "Running_Man",
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
            org_points.append([val[0], val[1]])
    return np.array(org_points)

import json
import random
from typing import List

import cv2
import numpy as np

from functions.funcs import points_are_close


class CameraData:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.json = self.load_json()
        # Extract the intrinsic and extrinsic parameters
        intrinsic = self.json["intrinsic"]
        extrinsic = self.json["extrinsic"]

        # Extract the individual parameters from intrinsic
        self.focal_length_x = intrinsic["focalLengthX"]
        self.focal_length_y = intrinsic["focalLengthY"]
        self.principal_point_x = intrinsic["principalPointX"]
        self.principal_point_y = intrinsic["principalPointY"]
        self.aspect_ratio = intrinsic["aspectRatio"]

        self.intrinsic_matrix = np.array(
            [
                [self.focal_length_x, 0, self.principal_point_x],
                [0, self.focal_length_y, self.principal_point_y],
                [0, 0, 1],
            ]
        )

        # Extract the individual parameters from extrinsic
        self.position = np.array(
            [
                extrinsic["position"]["x"],
                extrinsic["position"]["y"],
                extrinsic["position"]["z"],
            ]
        )
        rotation = extrinsic["rotation"]
        rvec = np.array([rotation["x"], rotation["y"], rotation["z"]])
        R, _ = cv2.Rodrigues(rvec)
        self.rotation = R

        self.extrinsic_matrix = np.array(
            [
                [R[0][0], R[0][1], R[0][2], self.position[0]],
                [R[1][0], R[1][1], R[1][2], self.position[1]],
                [R[2][0], R[2][1], R[2][2], self.position[2]],
                [0, 0, 0, 1],
            ]
        )

    def load_json(self):
        with open(self.json_path, "r") as f:
            return json.load(f)

    def transform_points_to_world(self, points: List[List[float]]):
        """Transforms points from camera coordinates to world coordinates"""
        return [self.transform_point_to_world(point) for point in points]

    def transform_point_to_world(self, point: List[float]):
        """Transforms a point from camera coordinates to world coordinates"""
        u = point[0]
        v = point[1]
        K = self.intrinsic_matrix
        extrinsic_matrix = self.extrinsic_matrix

        # convert the image point to normalized camera coordinates
        p_cam = np.dot(np.linalg.inv(K), np.array([u, v, 1]))

        # convert the normalized camera coordinates to world coordinates
        p_world_homogeneous = np.dot(
            np.linalg.inv(extrinsic_matrix), np.array([p_cam[0], p_cam[1], p_cam[2], 1])
        )

        # convert homogeneous coordinates to Cartesian coordinates
        p_world = p_world_homogeneous[:3] / p_world_homogeneous[3]
        return p_world

    def transform_points_to_camera(self, points: List[List[float]]):
        """Transforms points from world coordinates to camera coordinates"""
        return [self.transform_point_to_camera(point) for point in points]

    def transform_point_to_camera(self, point: List[float]):
        """Transforms a point from world coordinates to camera coordinates"""
        X = point[0]
        Y = point[1]
        Z = point[2]
        K = self.intrinsic_matrix
        extrinsic_matrix = self.extrinsic_matrix

        # convert the world point to normalized camera coordinates
        p_cam_homogeneous = np.dot(extrinsic_matrix, np.array([X, Y, Z, 1]))
        p_cam = p_cam_homogeneous[:3] / p_cam_homogeneous[3]

        # convert the normalized camera coordinates to image coordinates
        p_img_homogeneous = np.dot(K, p_cam)
        p_img = p_img_homogeneous[:2] / p_img_homogeneous[2]
        return p_img

    def test_valid(self):
        point = [random.randint(0, 640), random.randint(0, 480)]
        world_point = self.transform_point_to_world(point)
        og_point = self.transform_point_to_camera(world_point)
        if not points_are_close(point[0], point[1], og_point[0], og_point[1]):
            raise ValueError("Point transformation is not working")

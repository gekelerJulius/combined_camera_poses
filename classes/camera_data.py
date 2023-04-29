import json
import random
from typing import List

import cv2
import numpy as np

from functions.funcs import points_are_close


class CameraData:
    def __init__(self, fx, fy, cx, cy, aspect_ratio, position, rotation):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.aspect_ratio = aspect_ratio
        self.position = position
        self.rotation = rotation

        self.intrinsic_matrix = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

        R, _ = cv2.Rodrigues(rotation)
        self.R = R
        self.extrinsic_matrix = np.array(
            [
                [R[0][0], R[0][1], R[0][2], position[0]],
                [R[1][0], R[1][1], R[1][2], position[1]],
                [R[2][0], R[2][1], R[2][2], position[2]],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def create_from_json(json_path):
        # Extract the intrinsic and extrinsic parameters
        json_data = CameraData.load_json(json_path)
        intrinsic = json_data["intrinsic"]
        extrinsic = json_data["extrinsic"]

        # Extract the individual parameters from intrinsic
        fx = intrinsic["focalLengthX"]
        fy = intrinsic["focalLengthY"]
        cx = intrinsic["principalPointX"]
        cy = intrinsic["principalPointY"]
        aspect_ratio = intrinsic["aspectRatio"]

        # Extract the individual parameters from extrinsic
        position = np.array(
            [
                extrinsic["position"]["x"],
                extrinsic["position"]["y"],
                extrinsic["position"]["z"],
            ]
        )
        rotation = np.array(
            [
                extrinsic["rotation"]["x"],
                extrinsic["rotation"]["y"],
                extrinsic["rotation"]["z"],
            ]
        )

        return CameraData(fx, fy, cx, cy, aspect_ratio, position, rotation)

    @staticmethod
    def load_json(json_path):
        with open(json_path) as json_file:
            return json.load(json_file)

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

    def get_camera_matrix(self):
        return self.intrinsic_matrix

import json

import numpy as np
import cv2
from numpy import ndarray


class CameraData:
    def __init__(self, fx, fy, cx, cy, aspect_ratio, R, t):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.aspect_ratio = aspect_ratio

        self.intrinsic_matrix = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

        self.R = R
        self.t = t
        self.extrinsic_matrix3x4 = np.array(
            [
                [R[0][0], R[0][1], R[0][2], t[0]],
                [R[1][0], R[1][1], R[1][2], t[1]],
                [R[2][0], R[2][1], R[2][2], t[2]],
            ]
        )
        self.extrinsic_matrix4x4 = np.array(
            [
                [R[0][0], R[0][1], R[0][2], t[0]],
                [R[1][0], R[1][1], R[1][2], t[1]],
                [R[2][0], R[2][1], R[2][2], t[2]],
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
        fx = intrinsic["fx"]
        fy = intrinsic["fy"]
        cx = intrinsic["cx"]
        cy = intrinsic["cy"]
        width = intrinsic["width"]
        height = intrinsic["height"]
        aspect_ratio = width / height

        # Extract the individual parameters from extrinsic
        R = np.array(
            [
                [extrinsic["r00"], extrinsic["r01"], extrinsic["r02"]],
                [extrinsic["r10"], extrinsic["r11"], extrinsic["r12"]],
                [extrinsic["r20"], extrinsic["r21"], extrinsic["r22"]],
            ]
        )
        t = np.array([extrinsic["tx"], extrinsic["ty"], extrinsic["tz"]])

        return CameraData(fx, fy, cx, cy, aspect_ratio, R, t)

    @staticmethod
    def load_json(json_path):
        with open(json_path) as json_file:
            return json.load(json_file)

    def __str__(self):
        return f"CameraData(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, aspect_ratio={self.aspect_ratio}, R={self.R}, t={self.t})"

    def __repr__(self):
        return self.__str__()

    def transform_points_to_world(self, points: ndarray):
        """Transforms points from camera coordinates to world coordinates"""
        assert points.shape[1] == 2
        return [self.transform_point_to_world(point) for point in points]

    def transform_point_to_world(self, point: ndarray):
        """Transforms a point from camera coordinates to world coordinates"""
        assert point.shape == (2,)
        u = point[0]
        v = point[1]
        K = self.intrinsic_matrix
        extrinsic_matrix = self.extrinsic_matrix4x4

        # convert the image point to normalized camera coordinates
        p_cam = np.dot(np.linalg.inv(K), np.array([u, v, 1]))

        # convert the normalized camera coordinates to world coordinates
        p_world_homogeneous = np.dot(
            np.linalg.inv(extrinsic_matrix), np.array([p_cam[0], p_cam[1], p_cam[2], 1])
        )

        # convert homogeneous coordinates to Cartesian coordinates
        p_world = p_world_homogeneous[:3] / p_world_homogeneous[3]
        return p_world

    def transform_points_to_camera(self, points: ndarray) -> ndarray:
        """Transforms Nx3 points from world coordinates to Nx2 image coordinates"""
        assert len(points.shape) == 2
        assert points.shape[1] == 3
        return np.array([self.transform_point_to_camera(point) for point in points])

    def transform_point_to_camera(self, point: ndarray) -> ndarray:
        """Transforms a point from world coordinates to camera coordinates"""
        assert point.shape[0] == 3
        X = point[0]
        Y = point[1]
        Z = point[2]
        K = self.intrinsic_matrix
        extrinsic_matrix = self.extrinsic_matrix4x4

        # convert the world point to normalized camera coordinates
        p_cam_homogeneous = np.dot(extrinsic_matrix, np.array([X, Y, Z, 1]))
        p_cam = p_cam_homogeneous[:3] / p_cam_homogeneous[3]

        # convert the normalized camera coordinates to image coordinates
        p_img_homogeneous = np.dot(K, p_cam)
        p_img = p_img_homogeneous[:2] / p_img_homogeneous[2]
        return p_img

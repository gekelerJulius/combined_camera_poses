import json
from typing import Optional

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation


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
    def from_matrices(K, R, t) -> "CameraData":
        """
        Creates a CameraData object from the intrinsic and extrinsic matrices
        :param K: Intrinsic matrix (3x3)
        :param R: Rotation matrix (3x3)
        :param t: Translation vector (3x1) or (3,)
        """
        if t.shape == (3,):
            t = t.reshape((3, 1))
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        aspect_ratio = fx / fy
        tx = t[0][0]
        ty = t[1][0]
        tz = t[2][0]
        return CameraData(fx, fy, cx, cy, aspect_ratio, R, np.array([tx, ty, tz]))

    @staticmethod
    def create_from_json(json_path):
        # Extract the intrinsic and extrinsic parameters
        json_data = CameraData.load_json(json_path)
        intrinsic = json_data["intrinsic"]
        euler_unity = np.array(
            [
                json_data["eulerAngles"]["rx"],
                json_data["eulerAngles"]["ry"],
                json_data["eulerAngles"]["rz"],
            ]
        )
        # euler_cv = unity_to_cv_euler(euler_unity)
        # euler_cv = euler_unity
        t_unity = np.array(
            [
                json_data["position"]["tx"],
                json_data["position"]["ty"],
                json_data["position"]["tz"],
            ]
        )
        # t_cv = unity_to_cv(t_unity)
        t_cv = t_unity
        # R_cv = Rotation.from_euler("xyz", euler_cv).as_matrix()
        R_cv = Rotation.from_euler("xyz", euler_unity).as_matrix()

        # Extract the individual parameters from intrinsic
        fx = intrinsic["fx"]
        fy = intrinsic["fy"]
        cx = intrinsic["cx"]
        cy = intrinsic["cy"]
        width = intrinsic["width"]
        height = intrinsic["height"]
        aspect_ratio = width / height
        return CameraData(fx, fy, cx, cy, aspect_ratio, R_cv, t_cv)

    @staticmethod
    def load_json(json_path):
        with open(json_path) as json_file:
            return json.load(json_file)

    def __str__(self):
        return f"CameraData(fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, aspect_ratio={self.aspect_ratio}, R={self.R}, t={self.t})"

    def __repr__(self):
        return self.__str__()

    def get_projection_matrix(self):
        return np.dot(self.intrinsic_matrix, self.extrinsic_matrix3x4)

    def points_from_camera_to_world(self, points: ndarray) -> ndarray:
        """Transforms points from camera coordinates to world coordinates"""
        assert len(points.shape) == 2, f"points.shape = {points.shape}"
        assert points.shape[1] == 3, f"points.shape = {points.shape}"
        return np.array([self.point_from_camera_to_world(point) for point in points])

    def point_from_camera_to_world(self, point: ndarray) -> ndarray:
        """
        Transforms a point from camera coordinates to world coordinates
        """
        assert point.shape[0] == 3, f"point.shape = {point.shape}"
        X = point[0]
        Y = point[1]
        Z = point[2]
        world = np.dot(self.extrinsic_matrix4x4, np.array([X, Y, Z, 1]))
        world = world / world[3]
        return world[:3]

    def points_from_world_to_camera(self, points: ndarray) -> ndarray:
        """Transforms Nx3 points from world coordinates to Nx3 camera coordinates"""
        assert len(points.shape) == 2, f"points.shape = {points.shape}"
        assert points.shape[1] == 3, f"points.shape = {points.shape}"
        return np.array([self.point_from_world_to_camera(point) for point in points])

    def point_from_world_to_camera(self, point: ndarray) -> ndarray:
        """Transforms a point from world coordinates to camera coordinates"""
        assert point.shape[0] == 3, f"point.shape = {point.shape}"
        X = point[0]
        Y = point[1]
        Z = point[2]
        extrinsic_matrix = self.extrinsic_matrix4x4
        p_cam_homogeneous = np.dot(extrinsic_matrix, np.array([X, Y, Z, 1]))
        return p_cam_homogeneous[:3]

    def points_from_camera_to_image(self, points: ndarray) -> ndarray:
        """Transforms Nx3 points from camera coordinates to Nx2 image coordinates"""
        assert len(points.shape) == 2, f"points.shape = {points.shape}"
        assert points.shape[1] == 3, f"points.shape = {points.shape}"
        return np.array([self.point_from_camera_to_image(point) for point in points])

    def point_from_camera_to_image(self, point: ndarray) -> ndarray:
        """Transforms a point from camera coordinates to image coordinates"""
        assert point.shape[0] == 3, f"point.shape = {point.shape}"
        X = point[0]
        Y = point[1]
        Z = point[2]
        K = self.intrinsic_matrix
        p_cam_homogeneous = np.dot(K, np.array([X, Y, Z]))
        p_img = p_cam_homogeneous[:2] / p_cam_homogeneous[2]
        return p_img

    def points_from_image_to_camera(self, points: ndarray) -> ndarray:
        """Transforms Nx2 points from image coordinates to Nx3 camera coordinates"""
        assert len(points.shape) == 2, f"points.shape = {points.shape}"
        assert points.shape[1] == 2, f"points.shape = {points.shape}"
        return np.array([self.point_from_image_to_camera(point) for point in points])

    def point_from_image_to_camera(self, point: ndarray) -> ndarray:
        """Transforms a point from image coordinates to camera coordinates"""
        assert point.shape[0] == 2, f"point.shape = {point.shape}"
        u = point[0]
        v = point[1]
        K_inv = np.linalg.inv(self.intrinsic_matrix)
        p_cam_homogeneous = np.dot(K_inv, np.array([u, v, 1]))
        return p_cam_homogeneous

    def rotation_between_cameras(self, cam2_data: "CameraData") -> ndarray:
        """Calculates the rotation matrix between the two cameras"""
        return np.dot(self.R, np.linalg.inv(cam2_data.R))

    def translation_between_cameras(self, cam2_data: "CameraData") -> ndarray:
        """Calculates the translation vector between the two cameras as a (3x1) vector"""
        t = np.dot(self.R, cam2_data.t - self.t)
        return t.reshape((3, 1))


def unity_to_cv_euler(euler_unity: ndarray) -> ndarray:
    euler_cv = np.array(
        [euler_unity[0], euler_unity[2], euler_unity[1]]
    )  # swap Y and Z
    euler_cv[1:] = -euler_cv[1:]  # negate Y and Z
    return euler_cv

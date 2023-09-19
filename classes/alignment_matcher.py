from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from classes.person_history import PersonHistory
from functions.icp import do_icp_correl


class AlignmentMatcher:
    recorders: List[PersonHistory] = None

    def __init__(self, recorders: List[PersonHistory]):
        self.recorders = recorders

    def update_alignments(self, frame_num: int):
        # For now only support 2 recorders
        if len(self.recorders) != 2:
            return

        recorder1 = self.recorders[0]
        recorder2 = self.recorders[1]

        persons1 = recorder1.get_frame(frame_num)
        persons2 = recorder2.get_frame(frame_num)

        # Each recorder has an id and a list of persons per frame which have ids
        # We want each (right now the 2) recorders to store alignments between their persons using the ids

        cost_matrix = np.zeros((len(persons1), len(persons2)))
        for i in range(len(persons1)):
            for j in range(len(persons2)):
                cost_matrix[i, j] = self.get_cost(persons1[i], persons2[j])

        # Use Hungarian algorithm to find the best alignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        pairs = []
        for i, j in zip(row_ind, col_ind):
            pairs.append((persons1[i], persons2[j]))

        return pairs

    def get_cost(self, p1, p2):
        points1_3d = p1.get_pose_landmarks_numpy()
        points2_3d = p2.get_pose_landmarks_numpy()

        # Translate points to origin
        points1_3d = points1_3d - np.mean(points1_3d, axis=0)
        points2_3d = points2_3d - np.mean(points2_3d, axis=0)

        # Scale points to same size
        points1_3d = points1_3d / np.std(points1_3d, axis=0)
        points2_3d = points2_3d / np.std(points2_3d, axis=0)

        # 11: "left_shoulder",
        # 12: "right_shoulder",
        # Compare the distance between the shoulders for each person

        shoulder_dist1 = np.linalg.norm(points1_3d[11] - points1_3d[12])
        shoulder_dist2 = np.linalg.norm(points2_3d[11] - points2_3d[12])

        scale = shoulder_dist2 / shoulder_dist1
        points1_3d = points1_3d * scale

        # Find the best rotation to minimize the distance between the points
        transformation, rmse = do_icp_correl(points1_3d, points2_3d)
        return rmse

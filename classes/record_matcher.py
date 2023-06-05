import itertools
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy.optimize import linear_sum_assignment

from classes.PlotService import PlotService
from classes.camera_data import CameraData
from classes.logger import Logger
from classes.person import Person
from classes.person_recorder import PersonRecorder
from enums.logging_levels import LoggingLevel
from functions.calc_repr_errors import calc_reprojection_errors
from functions.estimate_extrinsic import estimate_extrinsic
from functions.funcs import diff_rotation_matrices, rotation_matrix_to_angles
from functions.get_person_pairs import compare_persons


class FrameRecord:
    frame_num: int
    persons1: List[Person]
    persons2: List[Person]
    cost_matrix: Dict[str, Dict[str, float]]
    estimated_person_pairs: List[Tuple[Person, Person]]
    estimated_extrinsic_matrix: Optional[np.ndarray]

    def __init__(
            self,
            frame_num: int,
            persons1: List[Person],
            persons2: List[Person],
            cost_matrix: Dict[str, Dict[str, float]],
    ):
        self.frame_num = frame_num
        self.persons1 = persons1
        self.persons2 = persons2
        self.cost_matrix = cost_matrix
        self.estimated_person_pairs = []
        self.estimated_extrinsic_matrix = None


class RecordMatcher:
    rec1: PersonRecorder
    rec2: PersonRecorder
    frame_records: List[FrameRecord]
    reprojection_errors: List[float]

    def __init__(self, rec1: PersonRecorder, rec2: PersonRecorder):
        self.rec1 = rec1
        self.rec2 = rec2
        self.frame_records = []
        self.reprojection_errors = []

    def get_frame_record(self, frame_num: int) -> Optional[FrameRecord]:
        for record in self.frame_records:
            if record.frame_num == frame_num:
                return record
        return None

    def get_all_previous_extrinsics(self, frame_num: int) -> List[np.ndarray]:
        res = []
        for record in self.frame_records:
            if record.frame_num < frame_num:
                res.append(record.estimated_extrinsic_matrix)
        return [x for x in res if x is not None]

    def get_alignment(self, frame_num: int, cam_data1: CameraData, cam_data2: CameraData) -> List[
        Tuple[Person, Person]]:
        look_back = 24  # 1 second for 24 fps
        recent_persons1 = self.rec1.get_recent_persons(frame_num, look_back)
        recent_persons2 = self.rec2.get_recent_persons(frame_num, look_back)
        if len(recent_persons1) == 0 or len(recent_persons2) == 0:
            return []

        # Look at all past records for now
        relevant_records = [
            x for x in self.frame_records if x.frame_num <= frame_num
        ]
        assert len(relevant_records) > 0, "No relevant records found"

        # possible_pairings = generate_unique_pairings(recent_persons1, recent_persons2)
        # best_pairing = None
        # best_pairing_cost = sys.maxsize
        # for pairing in possible_pairings:
        #     cost = 0
        #     # for record in relevant_records:
        #     #     for pair in subset:
        #     #         cost += record.cost_matrix[pair[0].name][pair[1].name]
        #
        #     lmks1 = []
        #     lmks2 = []
        #     for pair in pairing:
        #         crs = PersonRecorder.get_all_corresponding_frame_recordings(
        #             pair[0],
        #             pair[1],
        #             self.rec1,
        #             self.rec2,
        #             (frame_num - 24, frame_num),
        #             visibility_threshold=0.75,
        #         )
        #         if len(crs) == 0:
        #             continue
        #         lmks1.extend(crs[0])
        #         lmks2.extend(crs[1])
        #
        #     if len(lmks1) == 0 or len(lmks2) == 0:
        #         continue
        #
        #     pts1 = np.array([[lmk.x, lmk.y] for lmk in lmks1])
        #     pts2 = np.array([[lmk.x, lmk.y] for lmk in lmks2])
        #     err1, err2 = calc_reprojection_errors(pts1, pts2, cam_data1, cam_data2)
        #     cost += np.mean(err1) + np.mean(err2)
        #     if cost < best_pairing_cost:
        #         best_pairing = pairing
        #         best_pairing_cost = cost
        #
        # if best_pairing is None:
        #     return []
        #
        # self.get_frame_record(frame_num).estimated_person_pairs = best_pairing
        # return best_pairing

        cost_matrix = np.zeros((len(recent_persons1), len(recent_persons2)))

        for i, a in enumerate(recent_persons1):
            for j, b in enumerate(recent_persons2):
                costs = []
                for record in relevant_records:
                    has_person1 = False
                    has_person2 = False
                    for person in record.persons1:
                        if person.name == a.name:
                            has_person1 = True
                            break
                    for person in record.persons2:
                        if person.name == b.name:
                            has_person2 = True
                            break
                    if not has_person1 or not has_person2:
                        continue
                    matr_a = record.cost_matrix.get(a.name)
                    if matr_a is None:
                        continue
                    matr_b = matr_a.get(b.name)
                    if matr_b is None:
                        continue
                    costs.append(matr_b)
                cost_matrix[i, j] = np.mean(costs) if len(costs) > 0 else sys.maxsize

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pairs: List[Tuple[Person, Person]] = []
        for i, j in zip(row_ind, col_ind):
            for pair in pairs:
                if pair[0].name == recent_persons1[i].name:
                    print("Duplicate found")
                    exit(1)
                if pair[1].name == recent_persons2[j].name:
                    print("Duplicate found")
                    exit(1)
            pairs.append((recent_persons1[i], recent_persons2[j]))
        self.get_frame_record(frame_num).estimated_person_pairs = pairs
        return pairs

    def eval_frame(
            self,
            frame_num: int,
            img1=None,
            img2=None,
            cam_data1: CameraData = None,
            cam_data2: CameraData = None,
    ) -> None:
        prev_exts = self.get_all_previous_extrinsics(frame_num)
        estimated_extr = None
        extrs_length = len(prev_exts)
        if extrs_length >= 72:
            middle_index = int(extrs_length / 2)
            indexes_around_middle = [
                middle_index - 3,
                middle_index - 2,
                middle_index - 1,
                middle_index + 1,
                middle_index + 2,
                middle_index + 3,
            ]

            # Calculate average distance between middle index and other indexes
            diffs = []
            for i in indexes_around_middle:
                d = diff_rotation_matrices(
                    prev_exts[middle_index][:3, :3], prev_exts[i][:3, :3]
                )
                diffs.append(d)
            diff = float(np.mean(diffs))
            if diff < 5:
                estimated_extr = prev_exts[middle_index]
        persons1 = self.rec1.frame_dict.get(frame_num)
        persons2 = self.rec2.frame_dict.get(frame_num)

        if (
                persons1 is None
                or len(persons1) == 0
                or persons2 is None
                or len(persons2) == 0
        ):
            persons1 = []
            persons2 = []

        cost_matrix: Dict[str, Dict[str, float]] = {}
        for i in range(len(persons1)):
            a = persons1[i]
            cost_matrix[a.name] = {}
            matr = cost_matrix[a.name]
            for j in range(len(persons2)):
                b = persons2[j]
                diff = compare_persons(
                    a,
                    b,
                    img1,
                    img2,
                    self.rec1,
                    self.rec2,
                    frame_num,
                    cam_data1,
                    cam_data2,
                    estimated_extr=None,  # estimated_extr,
                )
                matr[b.name] = diff
        self.frame_records.append(
            FrameRecord(frame_num, persons1, persons2, cost_matrix)
        )

    def get_median_extrinsic(self, frame_num: int):
        extrinsics = self.get_all_previous_extrinsics(frame_num)
        if len(extrinsics) == 0:
            return None
        return np.median(extrinsics, axis=0)

    def estimate_extrinsic_matrix(
            self, frame_num: int, cam_data1: CameraData, cam_data2: CameraData
    ) -> None:
        frame_record = self.get_frame_record(frame_num)
        if frame_record is None:
            return None
        pairs = frame_record.estimated_person_pairs
        if pairs is None or len(pairs) == 0:
            return None

        points1_img = []
        points2_img = []
        for pair in pairs:
            points1_img.extend(pair[0].get_pose_landmarks_numpy_2d())
            points2_img.extend(pair[1].get_pose_landmarks_numpy_2d())

        points1_img = np.array(points1_img)
        points2_img = np.array(points2_img)
        frame_record.estimated_extrinsic_matrix = estimate_extrinsic(
            points1_img,
            points2_img,
            cam_data1.intrinsic_matrix,
            cam_data2.intrinsic_matrix,
        )
        R, t = (
            frame_record.estimated_extrinsic_matrix[:, :3],
            frame_record.estimated_extrinsic_matrix[:, 3],
        )
        prevs = self.get_all_previous_extrinsics(frame_num)
        if len(prevs) == 0:
            print("No previous extrinsics")
            return

        # Calculate median of previous extrinsics
        ext: ndarray = np.hstack((R, t.reshape(3, 1)))
        all_exts: List[ndarray] = [ext]
        all_exts.extend(prevs)
        # median = np.median(all_exts, axis=0)
        # median_R = median[:, :3]
        # median_t = median[:, 3]
        mean = np.mean(all_exts, axis=0)
        mean_R = mean[:, :3]
        mean_t = mean[:, 3]

        K1 = cam_data1.intrinsic_matrix
        K2 = cam_data2.intrinsic_matrix
        R1 = np.eye(3)
        R2 = mean_R
        t1 = np.zeros((3, 1))
        t2 = np.array([mean_t]).T
        est_cam_data1 = CameraData.from_matrices(K1, R1, t1)
        est_cam_data2 = CameraData.from_matrices(K2, R2, t2)

        print(f"Y-Axis angle: {rotation_matrix_to_angles(R2)[1]}")
        # points3d = triangulate_3d_points(
        #     points1_img,
        #     points2_img,
        #     est_cam_data1.get_projection_matrix(),
        #     est_cam_data2.get_projection_matrix()
        # )

        err1, err2 = calc_reprojection_errors(
            points1_img, points2_img, est_cam_data1, est_cam_data2
        )
        mean_err = float(np.mean(err1 + err2))
        print("Total reprojection error: ", mean_err)
        self.reprojection_errors.append(mean_err)

        plotter = PlotService.get_instance()
        err_plot: Figure = plotter.get_plot("reprojection_error")
        if err_plot is None:
            err_plot = plt.figure()
            axis: Axes = err_plot.add_subplot(111)
            axis.yaxis.axis_name = "Reprojection error"
            axis.xaxis.axis_name = "Frame number"
        else:
            axis: Axes = err_plot.axes[0]

        axis.clear()
        axis.plot(self.reprojection_errors, color="blue")
        plotter.add_plot(err_plot, "reprojection_error")
        # plt.pause(0.001)


def generate_subsets(pairs: List[Tuple[Person, Person]]) -> List[List[Tuple[Person, Person]]]:
    subsets = []
    for r in range(1, len(pairs) + 1):
        subsets.extend(itertools.combinations(pairs, r))
    return subsets


def generate_pairings_subsets(persons1: List[Person], persons2: List[Person]) -> List[List[Tuple[Person, Person]]]:
    pairs = list(itertools.product(persons1, persons2))
    return generate_subsets(pairs)


def generate_unique_pairings(list1: List[Person], list2: List[Person]) -> List[List[Tuple[Person, Person]]]:
    return [list(zip(list1, p)) for p in itertools.permutations(list2)]

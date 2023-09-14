import itertools
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy.optimize import linear_sum_assignment

from classes.camera_data import CameraData
from classes.person import Person
from classes.person_recorder import PersonRecorder
from classes.plot_service import PlotService
from functions.estimate_extrinsic import refine_extrinsic_estimation
from functions.get_person_pairs import compare_persons

matplotlib.use("TkAgg")


class FrameRecord:
    frame_num: int
    persons1: List[Person]
    persons2: List[Person]
    cost_matrix: Dict[str, Dict[str, float]]
    estimated_person_pairs: List[Tuple[Person, Person]]
    reprojection_error: Optional[float]

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
        self.reprojection_error = None


class RecordMatcher:
    recorders: List[PersonRecorder]
    frame_records: List[FrameRecord]
    extrinsics_estimation: Optional[ndarray]

    def __init__(self, recorders: List[PersonRecorder]):
        self.recorders = recorders
        self.frame_records = []
        self.extrinsics_estimation = None

    def get_frame_record(self, frame_num: int) -> Optional[FrameRecord]:
        for record in self.frame_records:
            if record.frame_num == frame_num:
                return record
        return None

    def get_records_until(self, frame_num: int) -> List[FrameRecord]:
        return [x for x in self.frame_records if x.frame_num <= frame_num]

    # def get_all_previous_extrinsics(self, frame_num: int) -> List[np.ndarray]:
    #     res = []
    #     for record in self.frame_records:
    #         if record.frame_num < frame_num:
    #             res.append(record.estimated_extrinsic_matrix)
    #     return [x for x in res if x is not None]

    def get_all_previous_reprojection_errors(self, frame_num: int) -> List[float]:
        res = []
        for record in self.frame_records:
            if record.frame_num < frame_num:
                res.append(record.reprojection_error)
        return res

    def get_alignment(
            self, frame_num: int, cam_data1: CameraData, cam_data2: CameraData
    ) -> List[Tuple[Person, Person]]:
        look_back = 6  # 0.25 seconds for 24 fps
        recent_persons_lists: List[List[Person]] = [
            rec.get_recent_persons(frame_num, look_back) for rec in self.recorders
        ]
        recent_persons_lengths = [len(x) for x in recent_persons_lists]
        # Check if any of the lists are empty
        if any(recent_persons_lengths) == 0:
            return []

        relevant_records = [x for x in self.frame_records if x.frame_num <= frame_num]
        assert len(relevant_records) > 0, "No relevant records found"

        recent_persons1 = recent_persons_lists[0]
        recent_persons2 = recent_persons_lists[1]

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
                    raw_cost = np.mean(costs) if len(costs) > 0 else -1

                    # Find out how many past frames the persons were paired
                    # If they were paired in previous frames, reduce the cost
                    # This is to avoid switching between persons too often
                    pairing_look_back = 12
                    last_pairs: List[List[Tuple[Person, Person]]] = [
                        x.estimated_person_pairs
                        for x in self.frame_records
                        if frame_num - pairing_look_back < x.frame_num < frame_num
                    ]
                    flattened_pairs: List[Tuple[Person, Person]] = list(
                        itertools.chain.from_iterable(last_pairs)
                    )
                    count = 0
                    for pair in flattened_pairs:
                        if pair[0].name == a.name and pair[1].name == b.name:
                            count += 1

                    pairing_score = 0.5
                    if len(last_pairs) > 0:
                        pairing_score = count / len(last_pairs)

                    cost_matrix[i, j] = raw_cost * (1 - (pairing_score * 1))
                    # cost_matrix[i, j] = raw_cost

        if len(cost_matrix) == 0:
            return []

        max_val = np.max(cost_matrix)
        if max_val == 0:
            return []

        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[i])):
                if cost_matrix[i, j] == -1:
                    cost_matrix[i, j] = max_val + 1

        cost_matrix = cost_matrix / np.max(cost_matrix)
        repr_error = 1000000
        pairs: List[Tuple[Person, Person]] = []
        limit: int = 3
        while repr_error > limit:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                for pair in pairs:
                    if pair[0].name == recent_persons1[i].name:
                        # The person is already paired
                        print("Duplicate found")
                        exit(1)
                    if pair[1].name == recent_persons2[j].name:
                        # The person is already paired
                        print("Duplicate found")
                        exit(1)
                pairs.append((recent_persons1[i], recent_persons2[j]))
            repr_error, est, pts3d = self.get_optimized_error(
                frame_num, cam_data1, cam_data2, pairs, False
            )
            if repr_error > limit:
                limit *= 1.25
                if limit > 30:
                    break
                pairs = []
                cost_matrix[row_ind, col_ind] *= 1.5
                repr_error = 1000000

        err, est, pts3d = self.get_optimized_error(
            frame_num, cam_data1, cam_data2, pairs, True
        )
        self.get_frame_record(frame_num).estimated_person_pairs = pairs
        self.get_frame_record(frame_num).reprojection_error = err
        self.get_frame_record(frame_num).estimated_extrinsics = est
        self.get_frame_record(frame_num).estimated_3d_points = pts3d
        print(f"Frame 6 {frame_num}: {len(pairs)} pairs, {err} error")
        print(pairs)
        return pairs

    def eval_frame(
            self,
            frame_num: int,
            img1=None,
            img2=None,
            cam_data1: CameraData = None,
            cam_data2: CameraData = None,
    ) -> None:
        # estimated_extr = calculate_weighted_extrinsic(
        #     self.get_all_previous_extrinsics(frame_num),
        #     self.get_all_previous_reprojection_errors(frame_num),
        # )
        estimated_extr = self.extrinsics_estimation
        estimation_confidence = 1
        rec1 = self.recorders[0]
        rec2 = self.recorders[1]
        persons1 = rec1.frame_dict.get(frame_num)
        persons2 = rec2.frame_dict.get(frame_num)
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
                    rec1,
                    rec2,
                    frame_num,
                    cam_data1,
                    cam_data2,
                    estimated_extr=estimated_extr,
                    estimation_confidence=estimation_confidence,
                )
                matr[b.name] = diff
        self.frame_records.append(
            FrameRecord(frame_num, persons1, persons2, cost_matrix)
        )

    def get_optimized_error(
            self,
            frame_num: int,
            cam_data1: CameraData,
            cam_data2: CameraData,
            pairs: Optional[List[Tuple[Person, Person]]] = None,
            update_plots=False,
    ) -> Optional[Tuple[float, ndarray, ndarray]]:
        rec1 = self.recorders[0]
        rec2 = self.recorders[1]
        frame_record = self.get_frame_record(frame_num)
        if frame_record is None:
            return None
        if pairs is None:
            pairs = frame_record.estimated_person_pairs
            if pairs is None or len(pairs) == 0:
                return None
        points1_img = []
        points2_img = []
        for person1, person2 in pairs:
            past_persons1 = rec1.get_frame_history(person1, (frame_num - 12, frame_num))
            past_persons2 = rec2.get_frame_history(person2, (frame_num - 12, frame_num))
            for i in range(len(past_persons1)):
                if past_persons1.get(i) is None or past_persons2.get(i) is None:
                    continue
                visible_indices: List[int] = Person.get_common_visible_landmark_indices(
                    past_persons1[i], past_persons2[i], 0.5
                )
                np1 = past_persons1[i].get_pose_landmarks_numpy_2d()[visible_indices]
                np2 = past_persons2[i].get_pose_landmarks_numpy_2d()[visible_indices]
                points1_img.extend(np1)
                points2_img.extend(np2)

            points1_img.extend(person1.get_pose_landmarks_numpy_2d())
            points2_img.extend(person2.get_pose_landmarks_numpy_2d())

        points1_img = np.array(points1_img)
        points2_img = np.array(points2_img)
        updated_estimation, points3d, error = refine_extrinsic_estimation(
            self.extrinsics_estimation,
            points1_img,
            points2_img,
            cam_data1.intrinsic_matrix,
            cam_data2.intrinsic_matrix,
        )
        if update_plots:
            plotter = PlotService.get_instance()
            err_plot: Figure = plotter.get_plot("reprojection_error")
            if err_plot is None:
                err_plot = plt.figure()
                axis: Axes = err_plot.add_subplot(111)
            else:
                axis: Axes = err_plot.axes[0]

            axis.clear()
            axis.yaxis.axis_name = "Reprojection error in pixels"
            axis.xaxis.axis_name = "Frame number"
            axis.set_xlabel(axis.xaxis.axis_name)
            axis.set_ylabel(axis.yaxis.axis_name)
            errs = [r.reprojection_error for r in self.frame_records if r is not None]
            axis.plot(
                errs,
                "b-",
            )
            plotter.add_plot(err_plot, "reprojection_error")

            # assert points3d.shape[0] == points1_img.shape[0]
            # points3d_persons = []
            # for i in range(0, points3d.shape[0]):
            #     if i % 33 == 0:
            #         points3d_persons.append([])
            #     points3d_persons[-1].append(points3d[i])
            #
            # scene_plot: Figure = plotter.get_plot("scene")
            # if scene_plot is None:
            #     scene_plot = plt.figure()
            #     axis: Axes = scene_plot.add_subplot(111, projection="3d")
            #     axis.yaxis.axis_name = "Y"
            #     axis.xaxis.axis_name = "X"
            #     axis.zaxis.axis_name = "Z"
            # else:
            #     axis: Axes = scene_plot.axes[0]
            # axis.clear()
            # for points3d_person in points3d_persons:
            #     points3d_person = np.array(points3d_person)
            #     plot_pose_3d(points3d_person, axis)
            # plotter.add_plot(scene_plot, "scene")
            # plt.pause(0.01)
        return error, updated_estimation, points3d

    # def get_extrinsic_estimation(self, frame_num: int) -> Optional[ndarray]:
    #     return calculate_weighted_extrinsic(
    #         self.get_all_previous_extrinsics(frame_num),
    #         self.get_all_previous_reprojection_errors(frame_num),
    #     )

    def report(self) -> None:
        if len(self.frame_records) == 0:
            return
        print(self.extrinsics_estimation)


def calculate_weighted_extrinsic(
        past_extrinsics: List[ndarray],
        reprojection_errors: List[float],
        decay_base: float = 0.5,
) -> Optional[ndarray]:
    """Calculate a weighted average of past camera extrinsics.
    Args:
        past_extrinsics (list of ndarray): List of past extrinsic matrices.
        reprojection_errors (list of float): List of reprojection errors for each past estimation.
        decay_base (float): Decay factor for weighting recent frames more heavily.
    Returns:
        np.array: Weighted average extrinsic matrix.
    """
    num_frames = len(past_extrinsics)
    weights = np.zeros(num_frames)

    if num_frames < 2:
        return None

    for i in range(num_frames):
        err = reprojection_errors[i]
        if err is None or err == 0:
            err = 10000
        error_weight = 1 / err
        n = num_frames - i - 1
        recency_weight = 1 / (decay_base * n + 1)
        weights[i] = error_weight * recency_weight

    weights = weights / np.sum(weights)
    weighted_extrinsic = np.zeros_like(past_extrinsics[0])
    for i in range(num_frames):
        weighted_extrinsic += weights[i] * past_extrinsics[i]
    return weighted_extrinsic


def generate_subsets(
        pairs: List[Tuple[Person, Person]]
) -> List[List[Tuple[Person, Person]]]:
    subsets = []
    for r in range(1, len(pairs) + 1):
        subsets.extend(itertools.combinations(pairs, r))
    return subsets


def generate_pairings_subsets(
        persons1: List[Person], persons2: List[Person]
) -> List[List[Tuple[Person, Person]]]:
    pairs = list(itertools.product(persons1, persons2))
    return generate_subsets(pairs)


def generate_unique_pairings(
        list1: List[Person], list2: List[Person]
) -> List[List[Tuple[Person, Person]]]:
    return [list(zip(list1, p)) for p in itertools.permutations(list2)]

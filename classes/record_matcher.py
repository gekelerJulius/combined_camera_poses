import itertools
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from sksurgerycore.algorithms.averagequaternions import weighted_average_quaternions

from classes.PlotService import PlotService
from classes.camera_data import CameraData
from classes.logger import Divider, Logger
from classes.person import Person
from classes.person_recorder import PersonRecorder
from enums.logging_levels import LoggingLevel
from functions.calc_repr_errors import calc_reprojection_errors
from functions.estimate_extrinsic import estimate_extrinsic
from functions.get_person_pairs import compare_persons


class FrameRecord:
    frame_num: int
    persons1: List[Person]
    persons2: List[Person]
    cost_matrix: Dict[str, Dict[str, float]]
    estimated_person_pairs: List[Tuple[Person, Person]]
    estimated_extrinsic_matrix: Optional[np.ndarray]
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
        self.estimated_extrinsic_matrix = None
        self.reprojection_error = None


class RecordMatcher:
    rec1: PersonRecorder
    rec2: PersonRecorder
    frame_records: List[FrameRecord]

    def __init__(self, rec1: PersonRecorder, rec2: PersonRecorder):
        self.rec1 = rec1
        self.rec2 = rec2
        self.frame_records = []

    def get_frame_record(self, frame_num: int) -> Optional[FrameRecord]:
        for record in self.frame_records:
            if record.frame_num == frame_num:
                return record
        return None

    def get_records_until(self, frame_num: int) -> List[FrameRecord]:
        return [x for x in self.frame_records if x.frame_num <= frame_num]

    def get_all_previous_extrinsics(self, frame_num: int) -> List[np.ndarray]:
        res = []
        for record in self.frame_records:
            if record.frame_num < frame_num:
                res.append(record.estimated_extrinsic_matrix)
        return [x for x in res if x is not None]

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
        recent_persons1 = self.rec1.get_recent_persons(frame_num, look_back)
        recent_persons2 = self.rec2.get_recent_persons(frame_num, look_back)
        if len(recent_persons1) == 0 or len(recent_persons2) == 0:
            return []

        relevant_records = [x for x in self.frame_records if x.frame_num <= frame_num]
        assert len(relevant_records) > 0, "No relevant records found"

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

        repr_error = 1000
        pairs: List[Tuple[Person, Person]] = []
        limit: int = 10
        while repr_error > limit:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                for pair in pairs:
                    assert pair[0].name != recent_persons1[i].name, "Duplicate found"
                    assert pair[1].name != recent_persons2[j].name, "Duplicate found"
                pairs.append((recent_persons1[i], recent_persons2[j]))

            mat, repr_error = self.get_extrinsic_and_repr_error(
                frame_num, cam_data1, cam_data2, pairs, False, False, False
            )
            if repr_error > limit:
                limit *= 1.1
                repr_error = 1000
                if limit > 150:
                    print("Limit exceeded")
                    break
                pairs = []
                cost_matrix[row_ind, col_ind] *= 1.3

        self.get_frame_record(frame_num).estimated_person_pairs = pairs
        if len(pairs) > 0:
            self.get_extrinsic_and_repr_error(
                frame_num, cam_data1, cam_data2, pairs, True, True, True
            )
            ext_and_err = self.get_extrinsic_and_repr_error(
                frame_num, cam_data1, cam_data2, None, False, False, False
            )
            if ext_and_err is None:
                return pairs
            mat, error = ext_and_err
            self.get_frame_record(frame_num).estimated_extrinsic_matrix = mat
            self.get_frame_record(frame_num).reprojection_error = error
        return pairs

    def eval_frame(
        self,
        frame_num: int,
        img1=None,
        img2=None,
        cam_data1: CameraData = None,
        cam_data2: CameraData = None,
    ) -> None:
        estimated_extr = calculate_weighted_extrinsic(
            self.get_all_previous_extrinsics(frame_num),
            self.get_all_previous_reprojection_errors(frame_num),
        )

        # estimated_extr = self.get_median_extrinsic(frame_num)

        prev_errs = self.get_all_previous_reprojection_errors(frame_num)
        prev_errs = [x for x in prev_errs if x is not None]
        if len(prev_errs) > 0:
            min_err = np.min(prev_errs)
        else:
            min_err = 1000
        # Max confidence is <= 1 and min confidence is >= 15
        confidence = 1 - (min_err / 15)
        if confidence < 0:
            confidence = 0
        if confidence > 1:
            confidence = 1
        estimation_confidence = confidence * 0.8

        # if len(self.frame_records) > 6:
        #     estimated_extr = calculate_weighted_extrinsic(
        #         self.get_all_previous_extrinsics(frame_num),
        #         self.get_all_previous_reprojection_errors(frame_num),
        #     )
        #     min_err = sys.maxsize
        #     min_err_index = None
        #     for i in range(len(self.frame_records)):
        #         err = self.frame_records[i].reprojection_error
        #         if err < min_err:
        #             min_err = err
        #             min_err_index = i
        #     if min_err < 30:
        #         indexes_between = list(range(min_err_index, len(self.frame_records)))
        #         mean_err = np.mean(
        #             [self.frame_records[i].reprojection_error for i in indexes_between]
        #         )
        #         max_err = np.max(
        #             [self.frame_records[i].reprojection_error for i in indexes_between]
        #         )
        #         if mean_err < max_err * 0.3:
        #             estimated_extr = self.get_extrinsic_estimation(frame_num)
        #             estimation_confidence = 1 - mean_err / (max_err * 0.3)

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
                    estimated_extr=estimated_extr,
                    estimation_confidence=estimation_confidence,
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

    def get_extrinsic_and_repr_error(
        self,
        frame_num: int,
        cam_data1: CameraData,
        cam_data2: CameraData,
        pairs: Optional[List[Tuple[Person, Person]]] = None,
        use_weighted_mean: bool = True,
        update_plots=False,
        print_stuff=False,
    ) -> Optional[Tuple[ndarray, float]]:
        frame_record = self.get_frame_record(frame_num)
        if frame_record is None:
            return None
        if pairs is None:
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
        ext_frame = estimate_extrinsic(
            points1_img,
            points2_img,
            cam_data1.intrinsic_matrix,
            cam_data2.intrinsic_matrix,
        )

        if ext_frame is None:
            return None

        R, t = (
            ext_frame[:, :3],
            ext_frame[:, 3],
        )

        ext = None
        if use_weighted_mean:
            ext = calculate_weighted_extrinsic(
                self.get_all_previous_extrinsics(frame_num),
                self.get_all_previous_reprojection_errors(frame_num),
            )

        if ext is None:
            ext = np.hstack([R, t[:, np.newaxis]])

        ext_R, ext_t = ext[:, :3], ext[:, 3]
        K1 = cam_data1.intrinsic_matrix
        K2 = cam_data2.intrinsic_matrix
        R1 = np.eye(3)
        R2 = ext_R
        t1 = np.zeros((3, 1))
        t2 = np.array([ext_t]).T
        est_cam_data1 = CameraData.from_matrices(K1, R1, t1)
        est_cam_data2 = CameraData.from_matrices(K2, R2, t2)

        err1, err2 = calc_reprojection_errors(
            points1_img, points2_img, est_cam_data1, est_cam_data2
        )
        mean_err = float(np.mean(err1 + err2))

        if print_stuff:
            Logger.log(mean_err, LoggingLevel.DEBUG, "mean_err")

        if update_plots:
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
            axis.set_ylim(0, 100)
            axis.set_xlim(0, frame_num)
            errs = [r.reprojection_error for r in self.frame_records]
            axis.plot(errs, "b-")
            plotter.add_plot(err_plot, "reprojection_error")

            # points3d = triangulate_3d_points(
            #     points1_img,
            #     points2_img,
            #     est_cam_data1.get_projection_matrix(),
            #     est_cam_data2.get_projection_matrix(),
            # )
            # assert points3d.shape[0] == points1_img.shape[0]
            #
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
            # axis.set_ylim3d(-1, 1)
            # axis.set_xlim3d(-1, 1)
            # axis.set_zlim3d(-1, 1)
            # for points3d_person in points3d_persons:
            #     points3d_person = np.array(points3d_person)
            #     plot_pose_3d(points3d_person, axis)
            # plotter.add_plot(scene_plot, "scene")
            plt.pause(0.01)
        return ext_frame, mean_err

    def get_extrinsic_estimation(self, frame_num: int) -> Optional[ndarray]:
        return calculate_weighted_extrinsic(
            self.get_all_previous_extrinsics(frame_num),
            self.get_all_previous_reprojection_errors(frame_num),
        )
        # past_records = self.get_records_until(frame_num)
        # min_error = sys.float_info.max
        # min_record = None
        # for record in past_records:
        #     if record.estimated_extrinsic_matrix is None:
        #         continue
        #     if record.reprojection_error < min_error:
        #         min_error = record.reprojection_error
        #         min_record = record
        # if min_record is None:
        #     return None
        # return min_record.estimated_extrinsic_matrix

    def report(self, cam_data1: CameraData, cam_data2: CameraData) -> None:
        true_R = cam_data2.R
        true_t = cam_data2.t
        est = self.get_extrinsic_estimation(self.frame_records[-1].frame_num)
        est_R = est[:, :3]
        est_t = est[:, 3]
        with Divider("Extrinsic estimation"):
            print(est_R)
            print(est_t)

        with Divider("True extrinsic"):
            print(true_R)
            print(true_t)


def calculate_weighted_extrinsic(
    past_extrinsics: List[ndarray],
    reprojection_errors: List[float],
    decay_base: float = 1,
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
        if reprojection_errors[i] is None:
            error_weight = 0
        else:
            error_weight = 1 / (reprojection_errors[i] ** 4)
        n = num_frames - i - 1
        recency_weight = 1 / (decay_base * n + 1)
        weights[i] = error_weight * recency_weight

    weights = weights / np.sum(weights)
    weighted_extrinsic = np.zeros_like(past_extrinsics[0])
    translations = [e[:3, 3] for e in past_extrinsics]
    rotations = [Rotation.from_matrix(e[:3, :3]) for e in past_extrinsics]
    quaternions = [r.as_quat() for r in rotations]

    # From (x, y, z, w) format to (w, x, y, z) format
    quaternions = np.array([np.array([q[3], q[0], q[1], q[2]]) for q in quaternions])
    avg_rotation = weighted_average_quaternions(quaternions, weights)

    # From (w, x, y, z) format to (x, y, z, w) format
    avg_rotation = np.array(
        [avg_rotation[1], avg_rotation[2], avg_rotation[3], avg_rotation[0]]
    )
    avg_rotation = Rotation.from_quat(avg_rotation)
    avg_translation = np.average(translations, axis=0, weights=weights)
    weighted_extrinsic[:3, :3] = avg_rotation.as_matrix()
    weighted_extrinsic[:3, 3] = avg_translation
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

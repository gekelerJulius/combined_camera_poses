from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import randomname
import seaborn as sns
from mediapipe.tasks.python.components.containers import Landmark
from numpy import ndarray
from scipy.optimize import linear_sum_assignment
from seaborn.palettes import _ColorPalette

from classes.person import Person


class PersonRecorder:
    frame_dict: Dict[int, List[Person]]
    name_dict: Dict[str, List[Person]]
    kalman_dict: Dict[str, Tuple[cv2.KalmanFilter, List[Person]]]
    kalman_prediction_dict: Dict[str, ndarray]
    id: str
    look_back: int
    color_palette: _ColorPalette
    palette_index: int

    def __init__(self, recorder_id: str, look_back: int = 12):
        self.id = recorder_id
        self.frame_dict = {}
        self.name_dict = {}
        self.kalman_dict = {}
        self.kalman_prediction_dict = {}
        self.look_back = look_back
        self.color_palette = sns.color_palette("dark", 6)
        self.palette_index = 0

    def add(self, persons: List[Person], frame_num: int, img=None):
        for person in persons:
            if self.frame_dict.get(person.frame_count) is None:
                self.frame_dict[person.frame_count] = []
            self.frame_dict.get(person.frame_count).append(person)
        self.update_kalman_filter(persons, frame_num, img)
        # self.plot_trajectories(img, frame_num)

    def get_recent_person_names(
            self, frame_num: int, look_back: int = None
    ) -> List[str]:
        if look_back is None:
            look_back = self.look_back
        recently_seen: List[Person] = []
        frames: List[int] = [frame_num - i for i in range(look_back)]

        for kalman_rec in self.kalman_dict.values():
            kalman_filter, persons = kalman_rec
            if persons is None or len(persons) == 0:
                continue
            last_person = persons[-1]
            if last_person.frame_count in frames:
                recently_seen.append(last_person)

        if recently_seen is None or len(recently_seen) == 0:
            return []
        return [r.name for r in recently_seen]

    def get_most_recent_record(self, name: str) -> Optional[Person]:
        kalman_rec = self.kalman_dict.get(name)
        if kalman_rec is None:
            return None
        kalman_filter, persons = kalman_rec
        if persons is None or len(persons) == 0:
            return None
        return persons[-1]

    def get_latest_frame_num_for_person(self, name: str) -> Optional[int]:
        max_frame = 0
        for frame_num, persons in self.frame_dict.items():
            for person in persons:
                if person.name == name:
                    max_frame = max(max_frame, frame_num)
        return max_frame if max_frame > 0 else None

    def first_record_persons(self, persons: List[Person]) -> None:
        for person in persons:
            name = randomname.get_name(adj="character")
            while self.name_dict.get(name) is not None:
                name = randomname.get_name(adj="character")
            person.name = name
            color = self.color_palette[self.palette_index]
            self.palette_index = (self.palette_index + 1) % len(self.color_palette)
            color = tuple([int(c * 255) for c in color])
            color = (int(color[0]), int(color[1]), int(color[2]), 0.5)
            person.color = color
            self.name_dict[name] = [person]
            self.init_kalman_filter(person)

    def init_kalman_filter(self, person: Person) -> None:
        centroid = person.centroid()
        kalman = PersonRecorder.create_kalman_filter(centroid)
        self.kalman_dict[person.name] = (kalman, [person])
        prediction = kalman.predict()
        self.kalman_prediction_dict[person.name] = prediction

    @staticmethod
    def create_kalman_filter(point: ndarray) -> cv2.KalmanFilter:
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kalman.statePre = np.array([[point[0]], [point[1]], [0], [0]], np.float32)
        return kalman

    def update_kalman_filter(
            self, persons: List[Person], frame_num: int, img=None
    ) -> None:
        predictions = [
            (name, pred) for name, pred in self.kalman_prediction_dict.items()
        ]
        matched: List[str] = []
        # Use hungarian algorithm to match predictions to persons
        if len(predictions) > 0:
            centroids = [person.centroid() for person in persons]
            cost_matrix = np.zeros((len(predictions), len(centroids)))
            for i, (name, prediction) in enumerate(predictions):
                last_centroid = self.kalman_dict[name][1][-1].centroid()
                last_x, last_y = last_centroid[0], last_centroid[1]
                x_pred, y_pred = prediction[0][0], prediction[1][0]
                vel_x_pred, vel_y_pred = prediction[2][0], prediction[3][0]

                for j, centroid in enumerate(centroids):
                    centroid = np.array([centroid[0], centroid[1]])
                    last_vel_x, last_vel_y = (centroid[0] - last_x), (
                            centroid[1] - last_y
                    )
                    # dist1 is distance between predicted centroid and current centroid
                    dist1 = np.linalg.norm(np.array([x_pred, y_pred]) - centroid)
                    # dist2 is distance between last centroid and current centroid
                    dist2 = np.linalg.norm(np.array([last_x, last_y]) - centroid)
                    # vel_dist is distance between predicted velocity and actual velocity
                    vel_dist = np.linalg.norm(
                        np.array([vel_x_pred, vel_y_pred])
                        - np.array([last_vel_x, last_vel_y])
                    )

                    # print("dist1: ", dist1)
                    # print("dist2: ", dist2)
                    # print("vel_dist: ", vel_dist)

                    dist = dist1 * (vel_dist ** 8)
                    cost_matrix[i, j] = dist

            if cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
                cost_matrix = cost_matrix / np.max(cost_matrix)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Update kalman filters with best matches
            for i, j in zip(row_ind, col_ind):
                cost = cost_matrix[i, j]
                if cost > 0.5:
                    continue
                name, prediction = predictions[i]
                assert prediction.shape == (4, 1)
                pos_prediction = np.array([prediction[0][0], prediction[1][0]])
                person = persons[j]
                c = person.centroid()
                pos_person = np.array([c[0], c[1]])
                # dist2 = np.linalg.norm(pos_prediction - pos_person)
                # if img is not None and dist2 > 20 and frame_num > 16:
                #     with Divider("Actual, Prediction, Distance"):
                #         print(pos_person)
                #         print(pos_prediction)
                #         print(dist2)
                #     copy = img.copy()
                #     cv2.circle(
                #         copy,
                #         (int(c[0]), int(c[1])),
                #         2,
                #         (0, 0, 255),
                #         -1,
                #     )
                #     cv2.circle(
                #         copy,
                #         (int(prediction[0][0]), int(prediction[1][0])),
                #         2,
                #         (255, 0, 0),
                #         -1,
                #     )
                #     cv2.imshow("Prediction", copy)
                #     cv2.waitKey(0)

                person.name = name
                person.color = self.name_dict[name][0].color
                matched.append(name)
                self.correct_and_predict_kalman(person.name, person.centroid())
                kalman, person_list = self.kalman_dict[name]
                person_list.append(person)

            self.first_record_persons(list(filter(lambda x: x.name is None, persons)))

        else:
            self.first_record_persons(persons)

        # Get all unmatched predictions
        unmatched_predictions = list(filter(lambda x: x[0] not in matched, predictions))
        for name, prediction in unmatched_predictions:
            last_frame_num = self.get_latest_frame_num_for_person(name)
            frames_since_last_seen = frame_num - last_frame_num
            if last_frame_num is None or frames_since_last_seen > 24 or frame_num < 24:
                self.kalman_dict.pop(name)
                self.kalman_prediction_dict.pop(name)
                assert self.kalman_dict.get(name) is None
                assert self.kalman_prediction_dict.get(name) is None
            else:
                # new_pred: ndarray = self.kalman_prediction_dict[name]
                # last_person_recorded: Optional[Person] = self.kalman_dict[name][1][-1]
                # last_centroid: ndarray = last_person_recorded.centroid()
                # last_x: float = last_centroid[0]
                # last_y: float = last_centroid[1]
                # x_pred: float = new_pred[0][0]
                # y_pred: float = new_pred[1][0]
                # centroid_diff_x: int = int(x_pred - last_x)
                # centroid_diff_y: int = int(y_pred - last_y)
                #
                # new_landmark_list: List[Landmark] = [
                #     Landmark(
                #         x=landmark.x + centroid_diff_x,
                #         y=landmark.y + centroid_diff_y,
                #         z=landmark.z,
                #         visibility=last_person_recorded.pose_landmarks[i].visibility,
                #     )
                #     for i, landmark in enumerate(last_person_recorded.pose_landmarks)
                # ]
                #
                # predicted_person: Person = Person(
                #     frame_count=frame_num,
                #     bounding_box=last_person_recorded.bounding_box.copy_moved(
                #         centroid_diff_x, centroid_diff_y
                #     ),
                #     pose_landmarks=new_landmark_list,
                #     pose_world_landmarks=[],
                # )
                # predicted_person.name = name
                # predicted_person.color = self.name_dict[name][0].color
                # self.kalman_dict[name][1].append(predicted_person)
                self.predict_kalman(name)

    def predict_kalman(self, name: str) -> None:
        kalman, person_list = self.kalman_dict[name]
        self.kalman_prediction_dict[name] = kalman.predict()

    def correct_kalman(self, name: str, centroid: ndarray) -> None:
        kalman, person_list = self.kalman_dict[name]
        kalman.correct(np.array([[np.float32(centroid[0])], [np.float32(centroid[1])]]))
        kalman_prediction = kalman.predict()
        self.kalman_prediction_dict[name] = kalman_prediction

    def correct_and_predict_kalman(
            self, name: str, centroid: Optional[ndarray]
    ) -> None:
        if centroid is not None:
            self.correct_kalman(name, centroid)
        self.predict_kalman(name)

    def plot_trajectories(self, img, frame_num=None) -> None:
        for name in self.kalman_dict:
            kalman, person_list = self.kalman_dict[name]
            if frame_num is not None:
                last_frame_num = self.get_latest_frame_num_for_person(name)
                if last_frame_num is None or frame_num < last_frame_num:
                    continue

            for i in range(1, len(person_list)):
                prev: Person = person_list[i - 1]
                after: Person = person_list[i]
                prev_np = prev.get_pose_landmarks_numpy()
                after_np = after.get_pose_landmarks_numpy()

                assert prev_np is not None
                assert after_np is not None
                assert prev_np.shape == (33, 3)
                assert after_np.shape == (33, 3)

                # l_wrist1 = prev_np[LANDMARK_INDICES_PER_NAME["left_wrist"]]
                # l_wrist2 = after_np[LANDMARK_INDICES_PER_NAME["left_wrist"]]
                #
                # cv2.line(
                #     img,
                #     (int(l_wrist1[0]), int(l_wrist1[1])),
                #     (int(l_wrist2[0]), int(l_wrist2[1])),
                #     prev.color,
                #     2,
                # )

                for j in range(1, 33):
                    # cv2.line(
                    #     img,
                    #     (int(prev_np[j, 0]), int(prev_np[j, 1])),
                    #     (int(after_np[j, 0]), int(after_np[j, 1])),
                    #     prev.color,
                    #     1,
                    # )
                    cv2.circle(
                        img,
                        (int(prev_np[j, 0]), int(prev_np[j, 1])),
                        1,
                        prev.color,
                        -1,
                    )

    def get_frame_history(
            self, p: Person, frame_range: Tuple[int, int] = (0, np.inf)
    ) -> Dict[int, Person]:
        all_records = self.kalman_dict[p.name][1]
        return {
            p.frame_count: p
            for p in all_records
            if frame_range[0] <= p.frame_count <= frame_range[1]
        }

    @staticmethod
    def get_all_corresponding_frame_recordings(
            p1: Person,
            p2: Person,
            recorder1: "Recorder",
            recorder2: "Recorder",
            frame_range: Tuple[int, int] = (0, np.inf),
            visibility_threshold: float = 0.5,
    ) -> Tuple[List[Landmark], List[Landmark], List[int]]:
        p1_rec_positions: Dict[int, Person] = recorder1.get_frame_history(
            p1, frame_range
        )
        p2_rec_positions: Dict[int, Person] = recorder2.get_frame_history(
            p2, frame_range
        )
        r1 = []
        r2 = []
        lmk_indexes = []

        common_keys = set(p1_rec_positions.keys()).intersection(
            set(p2_rec_positions.keys())
        )
        if len(common_keys) == 0:
            return [], [], []

        p1_positions: List[Person] = [p1_rec_positions[k] for k in common_keys]
        p2_positions: List[Person] = [p2_rec_positions[k] for k in common_keys]

        assert len(p1_positions) == len(p2_positions)

        for j in range(len(p1_positions)):
            pers1: Person = p1_positions[j]
            pers2: Person = p2_positions[j]

            common_indices: List[int] = Person.get_common_visible_landmark_indexes(
                pers1, pers2, visibility_threshold
            )
            lmks1: List[Landmark] = [lmk for lmk in pers1.get_pose_landmarks()]
            lmks1 = [lmks1[i] for i in common_indices]
            lmks2: List[Landmark] = [lmk for lmk in pers2.get_pose_landmarks()]
            lmks2 = [lmks2[i] for i in common_indices]
            r1.extend(lmks1)
            r2.extend(lmks2)
            lmk_indexes.extend(common_indices)
        return r1, r2, lmk_indexes

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

    def get_recent_persons(self, frame_num: int, look_back: int = None) -> List[Person]:
        if look_back is None:
            look_back = self.look_back
        recently_seen = []
        frames = [frame_num - i for i in range(look_back)]
        for frame in frames:
            if self.frame_dict.get(frame) is not None:
                persons = self.frame_dict.get(frame)
                for person in persons:
                    if person.name in [p.name for p in recently_seen]:
                        continue
                    recently_seen.append(person)

        if recently_seen is None or len(recently_seen) == 0:
            return []
        return [r for r in recently_seen]

    def get_trajectory(self, name: str) -> List[Person]:
        kalman, persons = self.kalman_dict.get(name)
        return persons

    def get_person_at_frame(self, name: str, frame_num: int) -> Optional[Person]:
        persons_in_frame = self.frame_dict.get(frame_num, [])
        for person in persons_in_frame:
            if person.name == name:
                return person
        return None

    def get_latest_frame_for_person(self, name: str) -> Optional[int]:
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
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )

        kalman.statePre = np.array([[centroid[0]], [centroid[1]], [0], [0]], np.float32)
        self.kalman_dict[person.name] = (kalman, [person])
        prediction = kalman.predict()
        self.kalman_prediction_dict[person.name] = prediction

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
                x_pred, y_pred = prediction[0], prediction[1]
                vel_x_pred, vel_y_pred = prediction[2], prediction[3]

                for j, centroid in enumerate(centroids):
                    centroid = np.array([centroid[0], centroid[1]])
                    last_vel_x, last_vel_y = (centroid[0] - last_x), (
                        centroid[1] - last_y
                    )
                    # dist1 is distance between predicted centroid and current centroid
                    dist1 = np.linalg.norm(np.array([x_pred, y_pred]) - centroid)
                    # dist2 is distance between last centroid and current centroid
                    dist2 = np.linalg.norm(np.array([last_x, last_y]) - centroid)
                    # vel_dist is distance between predicted velocity and last velocity
                    vel_dist = np.linalg.norm(
                        np.array([vel_x_pred, vel_y_pred])
                        - np.array([last_vel_x, last_vel_y])
                    )
                    dist = dist1 * (vel_dist**2)
                    cost_matrix[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Update kalman filters with best matches
            for i, j in zip(row_ind, col_ind):
                name, prediction = predictions[i]
                persons[j].name = name
                persons[j].color = self.name_dict[name][0].color
                matched.append(name)
                self.update_kalman_filter_single(
                    persons[j].name, persons[j].centroid(), img
                )
                kalman, person_list = self.kalman_dict[name]
                person_list.append(persons[j])

            # Add new persons
            self.first_record_persons(list(filter(lambda x: x.name is None, persons)))

        else:
            self.first_record_persons(persons)

        # Get all unmatched predictions
        unmatched_predictions = list(filter(lambda x: x[0] not in matched, predictions))
        for name, prediction in unmatched_predictions:
            last_frame = self.get_latest_frame_for_person(name)
            if last_frame is None or frame_num - last_frame > self.look_back:
                self.kalman_dict.pop(name)
                self.kalman_prediction_dict.pop(name)
                assert self.kalman_dict.get(name) is None
                assert self.kalman_prediction_dict.get(name) is None
            else:
                self.update_kalman_filter_single(name, prediction, img, correct=False)

    def update_kalman_filter_single(
        self, name: str, centroid: ndarray, img, correct=True
    ) -> None:
        kalman, person_list = self.kalman_dict[name]
        if correct:
            kalman.correct(
                np.array([[np.float32(centroid[0])], [np.float32(centroid[1])]])
            )
        kalman_prediction = kalman.predict()
        self.kalman_prediction_dict[name] = kalman_prediction

        # Plot person trajectory on image
        # for i in range(1, len(person_list)):
        #     prev: Person = person_list[i - 1]
        #     after: Person = person_list[i]
        #     prev_np = prev.get_pose_landmarks_numpy()
        #     after_np = after.get_pose_landmarks_numpy()
        #     for j in range(1, 33):
        #         cv2.line(
        #             img,
        #             (int(prev_np[j, 0]), int(prev_np[j, 1])),
        #             (int(after_np[j, 0]), int(after_np[j, 1])),
        #             prev.color,
        #             2,
        #         )

    def plot_trajectories(self, img) -> None:
        for name in self.kalman_dict:
            kalman, centroid_list = self.kalman_dict[name]
            centroid_list = np.array(centroid_list)

            # Plot centroid trajectory on image
            for i in range(1, len(centroid_list)):
                a = centroid_list[i - 1]
                b = centroid_list[i]
                cv2.line(
                    img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), 2
                )
        cv2.imshow(self.id, img)

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

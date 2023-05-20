import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import randomname
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.optimize import linear_sum_assignment

from classes.logger import Logger
from classes.person import Person
from enums.logging_levels import LoggingLevel


class PersonRecorder:
    frame_dict: Dict[int, List[Person]]
    name_dict: Dict[str, List[Person]]
    kalman_dict: Dict[str, Tuple[cv2.KalmanFilter, List[Person]]]
    kalman_prediction_dict: Dict[str, ndarray]
    id: str

    def __init__(self, id: str):
        self.id = id
        self.frame_dict = {}
        self.name_dict = {}
        self.kalman_dict = {}
        self.kalman_prediction_dict = {}

    def add(self, persons: List[Person], img=None):
        for person in persons:
            if self.frame_dict.get(person.frame_count) is None:
                self.frame_dict[person.frame_count] = []
            self.frame_dict.get(person.frame_count).append(person)
        self.update_kalman_filter(persons, img)

    def get(self, frame_count: int) -> List[Person]:
        return self.frame_dict.get(frame_count, [])

    def get_all(self) -> Dict[int, List[Person]]:
        return self.frame_dict

    # def eval(self, frame_count: int):
    #     persons = self.get(frame_count)
    #     previous_persons = self.get(frame_count - 1)
    #
    #     if len(persons) == 0:
    #         return
    #
    #     elif len(previous_persons) == 0:
    #         self.name_dict = {}
    #         self.kalman_dict = {}
    #         self.first_record_persons(persons)
    #
    #     else:
    #         distances = []
    #         for person in persons:
    #             for previous_person in previous_persons:
    #                 dist = person.pose_distance(previous_person)
    #                 distances.append((person, previous_person, dist))
    #
    #         distances.sort(key=lambda x: x[2])
    #         i = 0
    #         matched = []
    #         while i < len(distances):
    #             person, previous_person, dist = distances[i]
    #             person.name = previous_person.name
    #             person.color = previous_person.color
    #             self.name_dict.get(previous_person.name).append(person)
    #             self.update_kalman_filter(person)
    #             distances = list(filter(lambda x: x[0] != person and x[1] != previous_person, distances))
    #             matched.append(person)
    #         self.first_record_persons(list(filter(lambda x: x not in matched, persons)))

    def first_record_persons(self, persons: List[Person]) -> None:
        for person in persons:
            name = randomname.get_name(adj="character")
            person.name = name
            color = sns.color_palette("bright", 10)[random.randint(0, 9)]
            color = tuple([int(c * 255) for c in color])
            color = (int(color[0]), int(color[1]), int(color[2]), 0.5)
            person.color = color
            self.name_dict[name] = [person]
            self.init_kalman_filter(person)

    def init_kalman_filter(self, person: Person) -> None:
        centroid = person.centroid()
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)

        kalman.statePre = np.array([[centroid[0]], [centroid[1]], [0], [0]], np.float32)
        self.kalman_dict[person.name] = (kalman, [person])
        prediction = kalman.predict()
        self.kalman_prediction_dict[person.name] = prediction

    def update_kalman_filter(self, person: List[Person], img=None) -> None:
        # Get all predictions
        predictions = [(name, pred) for name, pred in self.kalman_prediction_dict.items()]
        matched: List[str] = []
        # Use hungarian algorithm to match predictions to persons
        if len(predictions) > 0:
            centroids = [person.centroid() for person in person]
            cost_matrix = np.zeros((len(predictions), len(centroids)))
            for i, (name, prediction) in enumerate(predictions):
                last_centroid = self.kalman_dict[name][1][-1].centroid()
                x_pred, y_pred = prediction[0], prediction[1]
                vel_x_pred, vel_y_pred = prediction[2], prediction[3]
                last_x, last_y = last_centroid[0], last_centroid[1]
                for j, centroid in enumerate(centroids):
                    centroid = np.array([centroid[0], centroid[1]])
                    last_vel_x, last_vel_y = (centroid[0] - last_x), (centroid[1] - last_y)
                    dist1 = np.linalg.norm(np.array([x_pred, y_pred]) - centroid)
                    dist2 = np.linalg.norm(np.array([last_x, last_y]) - centroid)
                    vel_dist = np.linalg.norm(np.array([vel_x_pred, vel_y_pred]) - np.array([last_vel_x, last_vel_y]))
                    dist = dist1 + dist2 + vel_dist
                    cost_matrix[i, j] = dist

            # Use hungarian algorithm to find best matches
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Update kalman filters with best matches
            for i, j in zip(row_ind, col_ind):
                name, prediction = predictions[i]
                person[j].name = name
                person[j].color = self.name_dict[name][0].color
                matched.append(name)
                self.update_kalman_filter_single(person[j].name, person[j].centroid(), img)
                kalman, person_list = self.kalman_dict[name]
                person_list.append(person[j])

            # Add new persons
            self.first_record_persons(list(filter(lambda x: x.name is None, person)))

        else:
            self.first_record_persons(person)

        # Get all unmatched predictions
        unmatched_predictions = list(filter(lambda x: x[0] not in matched, predictions))
        # Remove unmatched predictions that are too old
        # TODO

        # Predict further for unmatched predictions
        for name, prediction in unmatched_predictions:
            self.update_kalman_filter_single(name, prediction, img)

        cv2.imshow(self.id, img)
        cv2.waitKey(1)

    def update_kalman_filter_single(self, name: str, centroid: ndarray, img, correct=True) -> None:
        kalman, person_list = self.kalman_dict[name]
        if correct:
            kalman.correct(np.array([[np.float32(centroid[0])], [np.float32(centroid[1])]]))
        kalman_prediction = kalman.predict()
        self.kalman_prediction_dict[name] = kalman_prediction

        # Plot person trajectory on image
        for i in range(1, len(person_list)):
            prev: Person = person_list[i - 1]
            after: Person = person_list[i]
            prev_np = prev.get_pose_landmarks_numpy()
            after_np = after.get_pose_landmarks_numpy()
            for j in range(1, 33):
                cv2.line(img, (int(prev_np[j, 0]), int(prev_np[j, 1])), (int(after_np[j, 0]), int(after_np[j, 1])),
                         prev.color, 2)

    def plot_trajectories(self, img) -> None:
        for name in self.kalman_dict:
            kalman, centroid_list = self.kalman_dict[name]
            centroid_list = np.array(centroid_list)

            # Plot centroid trajectory on image
            for i in range(1, len(centroid_list)):
                a = centroid_list[i - 1]
                b = centroid_list[i]
                cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), 2)
        cv2.imshow(self.id, img)

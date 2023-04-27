from typing import List

import numpy as np
import cv2 as cv
from mediapipe.python.solutions.pose import PoseLandmark

from classes.logger import Logger
from classes.person import Person
from enums.logging_levels import LoggingLevel


def get_person_pairs(a: List[Person], b: List[Person], img1=None, img2=None):
    """
    Returns a list of tuples of Person objects, where each tuple contains two Person objects
    that are the same person in different images.
    """
    # Create a Record of all matches between people in a and b and their differences between each other
    # sort the Record by the smallest difference between people in a and b and return the sorted list
    record = set()

    for p1 in a:
        for p2 in b:
            sim1 = p1.get_landmark_sim(p2, img1, img2)
            sim2 = p2.get_landmark_sim(p1, img2, img1)

            if sim1 is None or sim2 is None:
                continue
            record.add((p1, p2, sim1, sim2))

    # Sort the record by the smallest difference between people in a and b
    def sort_key(record_item):
        return record_item[2] + record_item[3]

    sorted_record = sorted(record, key=sort_key)

    first_pair = sorted_record[0]

    # Show first pair for confirmation by user
    # if img1 is not None and img2 is not None:
    #     cv.imshow("First pair1", first_pair[0].draw(img1))
    #     cv.imshow("First pair2", first_pair[1].draw(img2))
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()

    proof_by_refutation(first_pair[0], first_pair[1], img1, img2)
    exit(0)

    # Take pairs from the beginning of the sorted record until there are no more people in a or b
    pairs = []
    used_a = set()
    used_b = set()

    for p1, p2, diff1, diff2 in sorted_record:
        if p1 in used_a or p2 in used_b:
            continue
        pairs.append((p1, p2))
        used_a.add(p1)
        used_b.add(p2)

    return pairs


def proof_by_refutation(person1: Person, person2: Person, img1, img2):
    """
    Returns True if the assumption pair is correct, False otherwise.
    """
    # If the assumption pair is correct, we can use the pose points as features to compare
    # the two images and see if they are the same person

    p1_landmarks = person1.get_pose_landmarks()
    p2_landmarks = person2.get_pose_landmarks()

    # right_hand1 = p1_landmarks[PoseLandmark.RIGHT_WRIST]
    # right_hand2 = p2_landmarks[PoseLandmark.RIGHT_WRIST]
    #
    # if right_hand1 is None or right_hand2 is None:
    #     return False
    #
    # r1 = np.array([right_hand1.x, right_hand1.y, right_hand1.z])
    # r2 = np.array([right_hand2.x, right_hand2.y, right_hand2.z])

    p1 = np.array([[]])
    p2 = np.array([[]])

    for i, lmk1 in enumerate(p1_landmarks):
        lmk2 = p2_landmarks[i]
        if (
            lmk1 is None
            or lmk2 is None
            or lmk1.visibility < 0.5
            or lmk2.visibility < 0.5
        ):
            continue

        point1 = np.array([lmk1.x, lmk1.y, lmk1.z])
        point2 = np.array([lmk2.x, lmk2.y, lmk2.z])

        if p1.size == 0:
            p1 = point1
            p2 = point2
        else:
            p1 = np.vstack((p1, point1))
            p2 = np.vstack((p2, point2))

    # Think about what can be done with the pose points to compare the two images

from typing import List

from classes.person import Person


def get_person_pairs(a: List[Person], b: List[Person], img_a=None, img_b=None):
    """
    Returns a list of tuples of Person objects, where each tuple contains two Person objects
    that are the same person in different images.
    """
    smallest_diff = 1000
    smallest_diff_person = None
    smallest_diff_rotation = None
    pairs1 = []
    pairs2 = []

    for person in a:
        for person2 in b:
            diff, rotation = person.get_landmark_diff(person2)

            if diff is None or rotation is None:
                continue

            if diff < smallest_diff:
                smallest_diff = diff
                smallest_diff_person = person2
                smallest_diff_rotation = rotation
        if smallest_diff_person is not None:
            pairs1.append((person, smallest_diff_person))

    smallest_diff = 1000
    smallest_diff_person = None
    smallest_diff_rotation = None
    for person in b:
        for person2 in a:
            diff, rotation = person.get_landmark_diff(person2)

            if diff is None or rotation is None:
                continue

            if diff < smallest_diff:
                smallest_diff = diff
                smallest_diff_person = person2
                smallest_diff_rotation = rotation
        if smallest_diff_person is not None:
            pairs2.append((person, smallest_diff_person))

    pairs = []
    for pair in pairs1:
        for pair2 in pairs2:
            if pair[0] == pair2[1] and pair[1] == pair2[0]:
                pairs.append(pair)

    print(pairs)
    return pairs

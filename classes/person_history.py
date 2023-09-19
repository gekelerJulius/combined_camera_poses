from typing import Dict, List, Optional, Tuple

import numpy as np

from classes.person import Person


class PersonHistory:
    frame_dict: Dict[int, List[Person]] = None
    look_back: int = 1

    def __init__(self):
        self.frame_dict = {}

    def add(self, person_list: List[Person], frame_count: int):
        self.frame_dict[frame_count] = person_list

    def get(self, frame_count: int) -> List[Person]:
        return self.frame_dict[frame_count]

    def get_recent_persons(self, frame_num: int, look_back: int = None) -> List[Person]:
        if look_back is None:
            look_back = self.look_back
        recently_seen = []
        frames = [frame_num - i for i in range(look_back)]
        for frame in frames:
            persons = self.frame_dict.get(frame, [])
            for person in persons:
                if person.name in [p.name for p in recently_seen]:
                    continue
                recently_seen.append(person)

        if recently_seen is None or len(recently_seen) == 0:
            return []
        return [r for r in recently_seen]

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

    def get_frame_history(
        self, p: Person, frame_range: Tuple[int, int] = (0, np.inf)
    ) -> Dict[int, Person]:
        history = {}
        for frame_num, persons in self.frame_dict.items():
            if frame_num < frame_range[0] or frame_num > frame_range[1]:
                continue
            for person in persons:
                if person.name == p.name:
                    history[frame_num] = person
        return history

    def get_common_frames_poses(
        self,
        other_history: "PersonHistory",
        p1_name: str,
        p2_name: str,
        frame_range: Tuple[int, int] = (0, np.inf),
    ) -> Tuple[List[int], List[int], List[int]]:
        p1_history = self.get_frame_history(
            self.get_person_at_frame(p1_name, frame_range[0]), frame_range
        )
        p2_history = other_history.get_frame_history(
            other_history.get_person_at_frame(p2_name, frame_range[0]), frame_range
        )
        common_frames = list(set(p1_history.keys()).intersection(p2_history.keys()))
        r1 = []
        r2 = []
        lmk_indices = []

        for j in range(len(common_frames)):
            frame_num = common_frames[j]
            p1 = p1_history[frame_num]
            p2 = p2_history[frame_num]
            indices = Person.get_common_visible_landmark_indices(p1, p2, 0.5)
            p1_landmarks = p1.pose_landmarks.landmark
            p2_landmarks = p2.pose_landmarks.landmark
            r1.extend([p1_landmarks[i] for i in indices])
            r2.extend([p2_landmarks[i] for i in indices])
            lmk_indices.extend(indices)

        return r1, r2, lmk_indices

    def get_frame(self, frame_num):
        return self.frame_dict.get(frame_num, [])

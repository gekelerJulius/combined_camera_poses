from typing import Dict, List

import randomname

from classes.person import Person


class PersonRecorder:
    frame_dict: Dict[int, List[Person]]
    name_dict: Dict[str, List[Person]]

    def __init__(self):
        self.frame_dict = {}
        self.name_dict = {}

    def add(self, person: Person):
        if self.frame_dict.get(person.frame_count) is None:
            self.frame_dict[person.frame_count] = []
        self.frame_dict.get(person.frame_count).append(person)

    def get(self, frame_count: int) -> List[Person]:
        return self.frame_dict.get(frame_count, [])

    def get_all(self) -> Dict[int, List[Person]]:
        return self.frame_dict

    def eval(self, frame_count: int):
        persons = self.get(frame_count)
        previous_persons = self.get(frame_count - 1)

        if len(persons) == 0:
            return

        elif len(previous_persons) == 0:
            self.name_dict = {}
            for person in persons:
                name = randomname.get_name(adj="character")
                person.name = name
                self.name_dict[name] = [person]

        else:
            distances = []
            for person in persons:
                for previous_person in previous_persons:
                    dist = person.pose_distance(previous_person)
                    distances.append((person, previous_person, dist))

            distances.sort(key=lambda x: x[2])
            i = 0
            matched = []
            try:
                while i < len(distances):
                    person, previous_person, dist = distances[i]
                    person.name = previous_person.name
                    self.name_dict.get(previous_person.name).append(person)
                    distances = list(filter(lambda x: x[0] != person and x[1] != previous_person, distances))
                    matched.append(person)
            except AttributeError as e:
                print(e)
                print(self.name_dict)
                print(distances[i])
                exit(1)

            for person in persons:
                if person not in matched:
                    name = randomname.get_name(adj="character")
                    person.name = name
                    self.name_dict[name] = [person]

import time

from classes.logger import Logger
from enums.logging_levels import LoggingLevel


class Timer:

    def __init__(self, name: str = "Timer"):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        Logger.log(f"{self.name} took {self.interval} seconds", LoggingLevel.INFO)

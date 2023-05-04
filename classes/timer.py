import time
import threading
from typing import Optional


class Timer:
    name: str
    start: float
    end: float
    interval: float
    thread: Optional[threading.Thread]

    def __init__(self, name: str = "Timer", interval: Optional[float] = 0.1):
        self.name = name
        self.interval = interval
        self.start = -1
        self.end = -1
        self.thread = None

    def __enter__(self):
        self.start = time.time()
        self.thread = threading.Thread(target=self.update_interval)
        self.thread.start()

    def elapsed(self):
        return time.time() - self.start

    def update_interval(self):
        while self.end < 0:
            print(f"\r{self.name} has been running for {self.elapsed()} seconds", end="")
            time.sleep(self.interval)

    def __exit__(self, *args):
        self.end = time.time()
        if self.thread:
            self.thread.join()
        print(f"\r{self.name} took {self.elapsed()} seconds")

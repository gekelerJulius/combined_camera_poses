from typing import Any

from enums.logging_levels import LoggingLevel

color_dict = {
    LoggingLevel.DEBUG: "\033[94m",
    LoggingLevel.INFO: "\033[92m",
    LoggingLevel.WARNING: "\033[93m",
    LoggingLevel.ERROR: "\033[91m",
    LoggingLevel.CRITICAL: "\033[91m",
}


class Logger:
    active_levels = [
        LoggingLevel.DEBUG,
        LoggingLevel.INFO,
        LoggingLevel.WARNING,
        LoggingLevel.ERROR,
        LoggingLevel.CRITICAL,
    ]

    def __init__(self):
        raise RuntimeError("Use static methods")

    @staticmethod
    def divider():
        print("#" * 80)

    @staticmethod
    def log(log_val: Any, level: LoggingLevel = LoggingLevel.INFO, label: str = None):
        if level not in Logger.active_levels:
            return
        message = str(log_val)
        print(f"{color_dict[level]}[{level.name}] {label + ': ' if label is not None else ''} \n{message}\033[0m")


class Divider:
    def __init__(self, label: str = None):
        self.label = label

    def __enter__(self):
        Logger.divider()
        if self.label is not None:
            Logger.log(self.label, LoggingLevel.INFO)

    def __exit__(self, exc_type, exc_val, exc_tb):
        Logger.divider()

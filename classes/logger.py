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
    def log(logVal: Any, level: LoggingLevel = LoggingLevel.INFO):
        if level not in Logger.active_levels:
            return

        if type(logVal) == str:
            message = logVal
        else:
            message = str(logVal)

        print(f"{color_dict[level]}[{level.name}] \n{message}\033[0m")

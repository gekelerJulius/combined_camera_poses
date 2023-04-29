from enums.logging_levels import LoggingLevel


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
    def log(level, message):
        if level not in Logger.active_levels:
            return

        color_dict = {
            LoggingLevel.DEBUG: "\033[94m",
            LoggingLevel.INFO: "\033[92m",
            LoggingLevel.WARNING: "\033[93m",
            LoggingLevel.ERROR: "\033[91m",
            LoggingLevel.CRITICAL: "\033[91m",
        }

        print(f"{color_dict[level]}[{level.name}] \n{message}\033[0m")

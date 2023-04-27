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
        print(f"[{level}]: {message}")

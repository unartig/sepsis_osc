import logging

import colorlog

from sepsis_osc.utils.config import log_file, cfg_log_level


def setup_logging(log_level: str = cfg_log_level, *, console_log: bool = True, log_file: str = log_file) -> None:
    for handler in logging.root.handlers[:]:  # Clean existing handlers
        logging.root.removeHandler(handler)
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "error":
        level = logging.ERROR
    else:
        level = logging.WARNING

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.propagate = False

    if console_log:
        console_formatter = colorlog.ColoredFormatter(
            "%(asctime)s - %(name)-30s - %(log_color)s%(levelname)-8s%(reset)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "magenta",
            },
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    if log_file:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

import logging

import colorlog

from sepsis_osc.utils.config import log_file, cfg_log_level


def setup_logging(log_level: str = cfg_log_level, console_log=True):
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "error":
        level = logging.ERROR
    else:
        level = logging.WARNING

    logger = logging.getLogger()
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

        logger.setLevel(level)
        logger.addHandler(console_handler)

    if log_file:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

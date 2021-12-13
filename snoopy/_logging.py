from logging import Logger

import colorlog

_handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    fmt="%(bold_blue)s[%(asctime)s]%(reset)s%(log_color)s %(levelname)-8s%(message)s "
        "%(reset)s%(bold_blue)s(%(module)s -> %(funcName)s on %(threadName)s)",
    datefmt="%H:%M:%S",
    reset=True,
    log_colors={
        'DEBUG': 'thin_white',
        'INFO': 'bold_green',
        'WARNING': 'bold_yellow',
        'ERROR': 'bold_red',
        'CRITICAL': 'bold_red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
_handler.setFormatter(formatter)
_global_log_level = "DEBUG"


def set_logging_level(log_level: str):
    global _global_log_level
    _global_log_level = log_level


# https://github.com/borntyping/python-colorlog/blob/master/doc/example.py
def get_logger(path: str) -> Logger:
    global _global_log_level
    logger = colorlog.getLogger(path.split(".")[-1] if "." in path else path)
    logger.setLevel(_global_log_level)
    logger.propagate = False
    logger.addHandler(_handler)

    return logger

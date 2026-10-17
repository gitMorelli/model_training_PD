"""Bridge between Python logging and an existing `write_log(str)` sink.

Every module in the package logs through `logging.getLogger(__name__)`, so all
of it lives under the "pd_training" logger. To keep sending those lines to the
file your old `write_log` function wrote to:

    from pd_training.logging_utils import attach_write_log
    attach_write_log(write_log)
"""
from __future__ import annotations

import logging
from typing import Callable

from lightning.pytorch.utilities import rank_zero_only

PACKAGE_LOGGER = "pd_training"


class WriteLogHandler(logging.Handler):
    def __init__(self, write_log: Callable[[str], None], level=logging.INFO):
        super().__init__(level)
        self.write_log = write_log

    def emit(self, record):
        try:
            self.write_log(self.format(record) + "\n")
        except Exception:  # never let logging crash training
            self.handleError(record)


def attach_write_log(write_log: Callable[[str], None], level=logging.INFO,
                     logger_name: str = PACKAGE_LOGGER) -> WriteLogHandler:
    handler = WriteLogHandler(write_log, level)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    logger.setLevel(min(logger.level or level, level))
    return handler


def rank_zero_logger(name: str):
    """(info, warning) functions that only emit on global rank 0."""
    logger = logging.getLogger(name)
    return rank_zero_only(logger.info), rank_zero_only(logger.warning)

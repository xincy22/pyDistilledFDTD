import logging
import sys
import os
from datetime import datetime
from typing import Dict, Tuple
from config.logger_config import LOG_FORMAT, LOG_DIR

class LoggerManager:
    _instance = None
    _initialized = False
    _loggers : Dict[str, logging.Logger] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not LoggerManager._initialized:
            LoggerManager._initialized = True
            os.makedirs(LOG_DIR, exist_ok=True)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        if '.' in name:
            name = name.split('.')[-1]

        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG)

            if not logger.handlers:
                file_handler, console_handler = cls._create_handlers(name)
                logger.addHandler(file_handler)
                logger.addHandler(console_handler)

            cls._loggers[name] = logger

        return cls._loggers[name]
    
    @classmethod
    def _create_handlers(cls, name: str) -> Tuple[logging.Handler, logging.Handler]:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(LOG_DIR, f"{name}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

        return file_handler, console_handler


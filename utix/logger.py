# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 

""" A colored logger with underline support """

import re
import logging

from colorama import Fore, init

# Initialize colorama
init(autoreset=True)


class Logger:
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA
    }

    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        # Only add handler if logger doesn't already have one
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(self.ColoredFormatter())
            self.logger.addHandler(handler)


    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            log_color = Logger.COLORS.get(record.levelname, Fore.WHITE)
            log_message = super().format(record)
            
            # Automatically underline any keyword argument values in the message
            if hasattr(record, 'underline_kwargs'):
                for key, value in record.underline_kwargs.items():
                    log_message = self.apply_underline(log_message, str(value), log_color)
            
            return f"{log_color}{log_message}{Logger.RESET}"

        @staticmethod
        def apply_underline(message, content, color):
            # Use ANSI escape code for underline and reset formatting
            underline_format = f"{Logger.UNDERLINE}{content}{Logger.RESET}{color}"
            return re.sub(re.escape(content), underline_format, message)


    def debug(self, msg, *args, **kwargs):
        self._log_with_underlining('debug', msg, *args, **kwargs)


    def info(self, msg, *args, **kwargs):
        self._log_with_underlining('info', msg, *args, **kwargs)


    def warning(self, msg, *args, **kwargs):
        self._log_with_underlining('warning', msg, *args, **kwargs)


    def error(self, msg, *args, **kwargs):
        self._log_with_underlining('error', msg, *args, **kwargs)


    def critical(self, msg, *args, **kwargs):
        self._log_with_underlining('critical', msg, *args, **kwargs)


    def _log_with_underlining(self, level, msg, *args, **kwargs):
        # Capture all keyword arguments for underlining
        underline_kwargs = kwargs
        # Add underline_kwargs to the LogRecord dynamically
        extra = {'underline_kwargs': underline_kwargs}
        # Format the message with keyword arguments, if any
        formatted_msg = msg.format(**underline_kwargs)
        formatted_msg = f"[{level.upper()}] {formatted_msg}"
        getattr(self.logger, level)(formatted_msg, extra=extra)


if __name__ == "__main__":

    logger = Logger(__name__)
    logger.info("This is a logger message with {name} and {action}.", name="test", action="underlined")
    logger.error("This is an error message.")

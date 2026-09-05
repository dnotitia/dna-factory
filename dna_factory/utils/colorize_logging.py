import logging
import os


# Color formatter for different log levels
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[37m",  # White
        "WARNING": "\033[33m",  # Yellow/Orange
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset color

    def format(self, record):
        # Get the original formatted message
        log_message = super().format(record)

        # Add color based on log level
        if record.levelname in self.COLORS:
            colored_levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
            # Replace the levelname in the formatted message
            log_message = log_message.replace(record.levelname, colored_levelname)

        return log_message


def format_logs_with_colors(variable_name):
    """Returns formatted environment variable with yellow color"""
    YELLOW = "\033[33m"  # Bright yellow color
    RESET = "\033[0m"
    value = os.environ.get(variable_name, "None")
    if value != "None" and value != "0":
        return f"{YELLOW}{variable_name}: {value}{RESET}"
    else:
        return f"{variable_name}: {value}"

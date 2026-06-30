"""
logger.py -- Structured Logging Setup for Federated Learning

Provides a factory function to create loggers with:
  - Console handler: INFO level, concise format
  - File handler: DEBUG level, detailed format with rotation
  - RotatingFileHandler: max 10MB per file, 5 backup files

Usage:
    from utils.logger import get_logger
    logger = get_logger("server")        # logs to logs/server.log
    logger = get_logger("client.hospital_a")  # logs to logs/client.log
"""

import os
import logging
from logging.handlers import RotatingFileHandler

# ============================================================
# Default logging configuration
# ============================================================
DEFAULT_LOG_DIR = "logs"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
DEFAULT_BACKUP_COUNT = 5              # Keep 5 rotated backups
DEFAULT_LOG_LEVEL = logging.DEBUG

# ============================================================
# Log format strings
# ============================================================
# Detailed format for file handler (includes timestamp, level, module)
FILE_FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Concise format for console handler
CONSOLE_FORMAT = "[%(levelname)-8s] [%(name)s] %(message)s"

# ============================================================
# Registry to prevent duplicate handlers on repeated calls
# ============================================================
_configured_loggers = set()


def get_logger(
    name: str,
    log_dir: str = DEFAULT_LOG_DIR,
    log_filename: str = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Create or retrieve a configured logger with console and file handlers.

    If the logger has already been configured (by a previous call), the
    existing logger is returned without adding duplicate handlers.

    Args:
        name: Logger name (e.g., "server", "client.hospital_a").
              Used as the logger identifier in log output.
        log_dir: Directory for log files (created if it doesn't exist).
        log_filename: Name of the log file. If None, defaults to
                      "{first_part_of_name}.log" (e.g., "server.log").
        console_level: Logging level for console output (default: INFO).
        file_level: Logging level for file output (default: DEBUG).

    Returns:
        A configured logging.Logger instance.
    """
    # --------------------------------------------------------
    # Return existing logger if already configured
    # --------------------------------------------------------
    if name in _configured_loggers:
        return logging.getLogger(name)

    # --------------------------------------------------------
    # Create logger instance
    # --------------------------------------------------------
    logger = logging.getLogger(name)
    logger.setLevel(DEFAULT_LOG_LEVEL)

    # Prevent log propagation to root logger (avoids duplicate output)
    logger.propagate = False

    # --------------------------------------------------------
    # Ensure log directory exists
    # --------------------------------------------------------
    os.makedirs(log_dir, exist_ok=True)

    # --------------------------------------------------------
    # Determine log file name
    # --------------------------------------------------------
    if log_filename is None:
        # Use the first component of the dotted name
        # e.g., "client.hospital_a" -> "client.log"
        base_name = name.split(".")[0]
        log_filename = f"{base_name}.log"

    log_path = os.path.join(log_dir, log_filename)

    # --------------------------------------------------------
    # File Handler — Detailed output with rotation
    # --------------------------------------------------------
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT)
    )
    logger.addHandler(file_handler)

    # --------------------------------------------------------
    # Console Handler — Concise output
    # --------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(CONSOLE_FORMAT)
    )
    logger.addHandler(console_handler)

    # --------------------------------------------------------
    # Register this logger to prevent duplicate configuration
    # --------------------------------------------------------
    _configured_loggers.add(name)

    logger.debug(f"Logger '{name}' initialized. File: {log_path}")

    return logger


def get_server_logger(log_dir: str = DEFAULT_LOG_DIR) -> logging.Logger:
    """
    Convenience function to create a server-specific logger.

    Logs are written to {log_dir}/server.log.

    Args:
        log_dir: Directory for log files.

    Returns:
        A configured logger named "server".
    """
    return get_logger(
        name="server",
        log_dir=log_dir,
        log_filename="server.log",
    )


def get_client_logger(
    hospital_id: str,
    log_dir: str = DEFAULT_LOG_DIR,
) -> logging.Logger:
    """
    Convenience function to create a client-specific logger.

    Logs are written to {log_dir}/client.log with the hospital_id
    included in the logger name for easy filtering.

    Args:
        hospital_id: Hospital identifier (e.g., "hospital_a").
        log_dir: Directory for log files.

    Returns:
        A configured logger named "client.{hospital_id}".
    """
    return get_logger(
        name=f"client.{hospital_id}",
        log_dir=log_dir,
        log_filename="client.log",
    )


# ============================================================
# CLI — Quick logger test
# ============================================================
if __name__ == "__main__":
    print("Testing logger setup...")

    # Test server logger
    server_log = get_server_logger()
    server_log.debug("This is a DEBUG message (file only)")
    server_log.info("This is an INFO message (console + file)")
    server_log.warning("This is a WARNING message")
    server_log.error("This is an ERROR message")

    # Test client logger
    client_log = get_client_logger("hospital_a")
    client_log.info("Hospital A client logger initialized")

    # Test duplicate prevention
    server_log_2 = get_server_logger()
    assert server_log is server_log_2, "Duplicate logger created!"
    print("Duplicate prevention: OK")

    print(f"\nLog files written to: {DEFAULT_LOG_DIR}/")
    print("Logger test complete.")

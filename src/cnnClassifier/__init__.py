"""Module provides logging functionality for the project."""
import os
import sys
import logging

# logging string
LOGGING_STRING = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# logs directory
LOGS_DIRECTORY = "logs"
logs_file_path = os.path.join(LOGS_DIRECTORY, "running_logs.log")
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_STRING,
    datefmt='%d-%b-%y %H:%M:%S',
    handlers=[
        logging.FileHandler(logs_file_path),
        logging.StreamHandler(sys.stdout)
    ]

)

logger = logging.getLogger("cnnClassifierLogger")

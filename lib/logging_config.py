# logging_config.py
import os
import time
import logging

os.environ['TZ']='US/Eastern'
time.tzset()
logger = logging.getLogger('fluid')


# Configure the logger
logger = logging.getLogger("my_package")  # Use the package name
logger.setLevel(logging.INFO)

# Define the handler and formatter
handler = logging.StreamHandler()  # Can also be FileHandler for file output
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S"
      )
handler.setFormatter(formatter)

# Avoid adding multiple handlers if the logger is imported multiple times
if not logger.hasHandlers():
    logger.addHandler(handler)
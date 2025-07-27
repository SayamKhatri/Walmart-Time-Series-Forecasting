import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_PATH = os.path.join(LOG_DIR, f"model_deployment_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("model_deployment")

logger.info("=" * 60)
logger.info("Model Deployment Logging System Initialized")
logger.info("=" * 60)
logger.info(f"Log file: {LOG_FILE_PATH}")
logger.info(f"Log level: {logging.getLevelName(logger.level)}")
logger.info("=" * 60) 
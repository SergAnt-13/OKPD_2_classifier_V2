from loguru import logger
from config.settings import LOGS_DIR

LOG_FILE = LOGS_DIR / "project.log"

logger.add(
    LOG_FILE,
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

def get_logger():
    return logger
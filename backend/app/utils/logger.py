import logging
import sys

# Configure stdout logging format
logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger instance with configured handlers and formatters.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger

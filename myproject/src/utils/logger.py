import logging
import os

def setup_logging(log_dir="logs"):
    """
    Set up logging to file and console.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")
import logging
import importlib.metadata
import importlib.util
import subprocess
from pathlib import Path
import sys
from typing import Tuple

def download_unzip_and_store(url, folder_path) -> None:
    """
    Use wget to download a ZIP file and unzip its contents to a specified folder.

    Args:
    - url (str): The URL of the ZIP file to download.
    - folder_path (str): The path of the folder where the ZIP file's contents will be extracted.
    # - folder_name (str): The name of the folder where the ZIP file's contents will be extracted.
    """

    import subprocess
    import os
    import zipfile

    # Ensure the destination folder exists
    if not os.path.exists(folder_path):  # directory exists?
        os.makedirs(folder_path)

        # Download the file with wget
        logging.info("Downloading the ZIP file with wget...")
        subprocess.run(["wget", "-O", "downloaded_file.zip", url])

        # Unzipping the file
        logging.info("Unzipping the file...")
        with zipfile.ZipFile("downloaded_file.zip", "r") as zip_ref:
            zip_ref.extractall(folder_path)

        # Clean up (delete) the zip file after extraction
        os.remove("downloaded_file.zip")
        logging.info(f"File successfully downloaded and extracted to '{folder_path}'")

    else:
        logging.info(f"The data already exists at '{folder_path}'")

class TeeStream:
    """Tees a stream to both terminal and a file, with no formatting overhead."""
    def __init__(self, original_stream, file_path, mode="w"):
        self.original_stream = original_stream
        self.file = open(file_path, mode, encoding="utf-8", buffering=1)  # line-buffered

    def write(self, message):
        self.original_stream.write(message)
        self.file.write(message)

    def flush(self):
        self.original_stream.flush()
        self.file.flush()

    def isatty(self):
        return False

    def close(self):
        self.file.close()

def add_logger(cfg, state) -> None:
    
    logger = logging.getLogger("igm")
    logger.setLevel(cfg.core.igm_logging_level)

    # Prevent messages from propagating to Hydra's root logger (avoids duplicates)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(cfg.core.igm_logging_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — Hydra already set cwd to the output dir, so this lands there
    file_handler = logging.FileHandler("igm.log", mode="w", encoding="utf-8")
    file_handler.setLevel(cfg.core.igm_logging_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    state.logger = logger
    
    # Tee stdout/stderr directly to file - no logger overhead
    sys.stdout = TeeStream(sys.__stdout__, "terminal.log")
    sys.stderr = TeeStream(sys.__stderr__, "terminal.log", mode="a")  # append so both go to same file

def get_igm_version() -> Tuple[str, str]:
    try:
        version = importlib.metadata.version("igm")
        source = 'PyPi'
        return (source, version)
    except importlib.metadata.PackageNotFoundError:
        pass

    try:
        igm_root = Path(__file__).resolve().parent
        hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=igm_root,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        source = 'git'
        return (source, hash)
    except Exception:
        pass

    return ("unknown", "unknown")

def write_igm_version(output_dir: Path) -> None:
    source, version = get_igm_version()
    with open(output_dir / "version.txt", "a", encoding="utf-8") as f:
        f.write(f"source: {source}\n")
        f.write(f"version: {version}\n")
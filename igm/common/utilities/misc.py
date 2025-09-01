import logging


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


def add_logger(cfg, state) -> None:

    # ! Ignore logging file for now...
    # if cfg.logging_file == "":
    #     pathf = ""
    # else:
    #     pathf = cfg.logging_file

    logging.basicConfig(
        # filename=pathf,
        encoding="utf-8",
        filemode="w",
        level=cfg.core.logging_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.root.setLevel(cfg.core.logging_level)

    state.logger = logging.getLogger("igm")

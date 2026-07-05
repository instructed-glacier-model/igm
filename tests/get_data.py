#!/usr/bin/env python3
"""
Download ISMIP-HOM test data for IGM solver tests.
Cross-platform script that works on Linux, Mac, and Windows.
"""

import os
import shutil
import sys
import tempfile
import zipfile
from urllib.request import urlopen, Request

ISMIP_HOM_OGA_URL = "https://frank.pattyn.web.ulb.be/ismip/tc-2-95-2008-supplement.zip"
ISMIP_HOM_AROLLA_URL = "https://frank.pattyn.web.ulb.be/ismip/arolla100.dat"
ISMIP_HOM_OGA_ZIP_INNER = "tc-2007-0019-sp2.zip"
ISMIP_HOM_OGA_ZIP_OUTER = "tc-2-95-2008-supplement.zip"
ISMIP_HOM_TARGET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_iceflow", "ismip_hom", "data")


def download_file(url: str, dest_path: str) -> None:
    """Download a file from URL to destination path."""
    print(f"Downloading {url}...")
    try:
        # Add a user agent to avoid potential 403 errors
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            with open(dest_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        raise


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a zip file to the specified directory."""
    print(f"Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def main():
    """Main function to download and extract ISMIP-HOM data."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Using temp dir: {tmpdir}")

        # Download ISMIP-HOM OGA reference data
        outer_zip_path = os.path.join(tmpdir, ISMIP_HOM_OGA_ZIP_OUTER)
        download_file(ISMIP_HOM_OGA_URL, outer_zip_path)

        # Extract outer zip
        extract_zip(outer_zip_path, tmpdir)

        # Extract inner zip
        inner_zip_path = os.path.join(tmpdir, ISMIP_HOM_OGA_ZIP_INNER)
        extract_zip(inner_zip_path, tmpdir)

        # Create target directory and copy OGA data
        oga_target = os.path.join(ISMIP_HOM_TARGET_DIR, "oga")
        os.makedirs(ISMIP_HOM_TARGET_DIR, exist_ok=True)

        # Remove existing oga directory if it exists
        if os.path.exists(oga_target):
            shutil.rmtree(oga_target)

        # Copy the extracted data
        oga_source = os.path.join(tmpdir, "ismip_all", "oga")
        shutil.copytree(oga_source, oga_target)

        print(
            f"✅ ISMIP-HOM OGA reference data downloaded to {os.path.abspath(oga_target)}"
        )

    # Download ISMIP-HOM Arolla input data
    arolla_dir = os.path.join(ISMIP_HOM_TARGET_DIR, "arolla")
    os.makedirs(arolla_dir, exist_ok=True)

    arolla_file = os.path.join(arolla_dir, "arolla100.dat")
    download_file(ISMIP_HOM_AROLLA_URL, arolla_file)

    print(f"✅ ISMIP-HOM Arolla input data downloaded to {os.path.abspath(arolla_dir)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

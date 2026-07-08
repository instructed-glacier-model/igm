#!/usr/bin/env python
# Copyright (C) 2021-2022 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3)

from setuptools import setup, find_packages
import os


def package_files(directory):
    paths = []
    for path, __, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="igm-model",
    version="3.2.0",
    author="IGM authors",
    author_email="guillaume.jouvet@unil.ch",
    url="https://igm-model.org/",
    license="gpl-3.0",
    packages=find_packages(include=["igm", "igm.*"]),
    include_package_data=True,
    package_data={"igm": package_files("igm/emulators")},
    entry_points={"console_scripts": ["igm_run = igm.igm_run:main"]},
    description="IGM - a glacier evolution model",
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires=">=3.10,<3.13",
    install_requires=[
        "tensorflow[and-cuda]==2.17.0",
        # Explicit numpy bound: TF requires it anyway, but stating it here makes
        # pip refuse later upgrades to numpy 2.x that would break the install.
        "numpy>=1.26,<2.0",
        "matplotlib",
        "scipy",
        "netCDF4>=1.6.5",
        "xarray",
        "rasterio",
        "pyproj",
        "geopandas",
        "oggm",
        "salem",
        "pyyaml",
        "importlib_resources",
        "tqdm",
        "hydra-core",
        "omegaconf",
        "nvtx",
        "typeguard",
        "rich",
        "optuna>=3.0",
    ],
    extras_require={
        "dev": ["pytest"],
        "vtk": ["pyvista"],  # optional VTP output in data_assimilation
    },
)

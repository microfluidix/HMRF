#!/usr/bin/env python

from setuptools import setup

setup(
    name="biograph",
    version="0.1",
    description="Extracting single-cell data from segmented images",
    author="Gustave Ronteix, Valentin Bonnet",
    author_email="gustave.ronteix@pasteur.fr",
    url="",
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "pytest",
        "tqdm",
        "pandas",
        "pip",
        "scikit-image",
        "matplotlib",
        "sklearn",
        "tox",
        "pytest",
    ],
    python_requires=">=3.8",
    packages=["biograph", "biograph.graphprops"],
)

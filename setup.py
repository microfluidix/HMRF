#!/usr/bin/env python

from setuptools import setup

setup(
    name="biograph",
    version="0.1",
    description="Extracting single-cell data from segmented images",
    author="Valentin Bonnet, Gustave Ronteix",
    author_email="gustave.ronteix@pasteur.fr",
    url="",
    install_requires=[
        "numpy",
        "tifffile",
        "scipy",
        "networkx",
        "pytest",
        "tqdm",
        "pandas",
        "pip",
        "scikit-image",
        "matplotlib",
        "black",
        "sklearn"
    ],
    packages = ['biograph', 
                'biograph.graphprops']
)

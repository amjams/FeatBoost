#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="featboost",
    version="1.0",
    packages=find_packages(include=["featboost", "featboost.*"]),
    author="Ahmad Alsahaf",
    install_requires=[
        "numpy==1.20.2",
        "scikit-learn==0.24.2",
    ],
)

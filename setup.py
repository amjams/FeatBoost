#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="featboost",
    version="1.1",
    packages=find_packages(include=["featboost", "featboost.*"]),
    author="Ahmad Alsahaf",
    install_requires=[
        "numpy>=1.19",
        "scikit-learn>=0.24",
    ],
    setup_requires=["black==21.4b2", "pytest-runner==5.3.0"],
    tests_require=["pytest==6.2.3"],
)

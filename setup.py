#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='nnvision',
    version='0.0.0',
    description='Envisioning the biological visual system with DNN',
    author='Konstantin Willeke',
    author_email='konstantin.willeke@gmail.com',
    packages=find_packages(exclude=[]),
    install_requires=[],
)

#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='nnvision',
    version='0.1',
    description='Envisioning the biological visual system with DNN',
    author='Konstantin Willeke',
    author_email='konstantin.willeke@gmail.com',
    packages=find_packages(exclude=[]),
    install_requires=[
        "einops",
        "scikit-image==0.19.1",
        "numpy==1.22.0",
        'nnfabrik',
        'neuralpredictors @ git+https://github.com/KonstantinWilleke/neuralpredictors.git@transformer_readout'
    ],
)


"""
Setup file for the project."""

from setuptools import setup, find_packages

setup(
    name="generate_digits",
    version="1.0",
    packages=find_packages(),
    author="pranshu-raj-211",
    install_requires=[
        "numpy",
        "tensorflow",
        "Pillow",
        "fastapi",
    ],
)

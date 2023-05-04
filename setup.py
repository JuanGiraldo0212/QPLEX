from setuptools import setup, find_packages

version = "0.1.0"

with open("requirements.txt") as reqs_file:
    required_packages = reqs_file.read().splitlines()

setup(
    name="QPLEX",
    version=version,
    author="Juan Giraldo, José Ossorio",
    author_email="juanfer021299@gmail.com",
    description="A Python library for modelling and solving optimization problems using classical and quantum resources",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JuanGiraldo0212/QPLEX",
    packages=find_packages(),
    install_requires=required_packages
)

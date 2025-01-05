from setuptools import setup, find_packages

setup(
    name="sighthound",
    version="1.0.0",
    description="A Python package for fusing GPS data, triangulation, and generating probability density functions.",
    author="Your Name",
    author_email="your_email@example.com",
    packages=find_packages(),
    install_requires=[
        "filterpy==1.4.5",
        "numpy==1.23.5",
        "pandas==1.5.3",
        "requests==2.28.2",
        "gpxpy==1.5.0",
        "fitparse==1.2.0",
        "geopy==2.3.0",
        "dubins"
    ],
    python_requires=">=3.8",
)

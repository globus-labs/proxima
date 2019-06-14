import os
from setuptools import setup, find_packages

# single source of truth for package version
version_ns = {}
with open(os.path.join("proxima", "version.py")) as f:
    exec(f.read(), version_ns)
version = version_ns['__version__']

setup(
    name='proxima',
    version=version,
    packages=find_packages(),
    description='System for learning approximations for and replacing expensive function calls on-the-fly',
    install_requires=[],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
    ],
    author='Logan Ward',
    author_email='lward@anl.gov',
    license="Apache License, Version 2.0",
    url="https://github.com/globus-labs"
)

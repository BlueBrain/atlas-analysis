#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

VERSION = imp.load_source("", "atlas_analysis/version.py").__version__

setup(
    name="atlas-analysis",
    author="BlueBrain NSE",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="Library containing atlas analyses for circuit building",
    url="https://bbpteam.epfl.ch/project/issues/projects/NSETM/issues",
    download_url="ssh://bbpcode.epfl.ch/nse/atlas-analysis",
    license="BBP-internal-confidential",
    install_requires=[
        'click>=7.0',
        'geomdl>=5.2.8',
        'lazy>=1.4',
        'numpy>=1.16.3',
        'networkx>=2.3',
        'pathos>=0.2.3',
        'plotly-helper>=0.0.2',
        'pyquaternion>=0.9.5',
        'scipy>=1.2.1',
        'voxcell>=2.6.2',
        'vtk>=8.1.2',
        'six>=1.12.0',
        'scikit-image>=0.16.1'
    ],
    packages=find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    entry_points={
        'console_scripts': [
            'atlas-analysis=atlas_analysis.app.__main__:main'
        ]
    }
)

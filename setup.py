#!/usr/bin/env python
import importlib.util

from setuptools import setup, find_packages


spec = importlib.util.spec_from_file_location(
    "atlas_analysis.version",
    "atlas_analysis/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.__version__

setup(
    name="atlas-analysis",
    author="BlueBrain NSE",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    version=VERSION,
    description="Library containing atlas analyses for circuit building",
    long_description="Library containing atlas analyses for circuit building",
    long_description_content_type="text/plain",
    url="https://bbpteam.epfl.ch/project/issues/projects/NSETM/issues",
    download_url="https://bbpgitlab.epfl.ch/nse/atlas-analysis",
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
    python_requires=">=3.7",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points={
        'console_scripts': [
            'atlas-analysis=atlas_analysis.app.__main__:main'
        ]
    }
)

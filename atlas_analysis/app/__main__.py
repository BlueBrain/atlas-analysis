"""
Collection of tools for atlas analysis
"""

import logging
import click

from atlas_analysis.app import atlas, meshes, curation, planes
from atlas_analysis.version import VERSION


def main():
    """ Collection of tools for atlas analysis"""
    logging.basicConfig(level=logging.INFO)
    app = click.Group('atlas_analysis', {
        'atlas': atlas.app,
        'mesh': meshes.app,
        'curation': curation.app,
        'planes': planes.app,
    })
    app = click.version_option(VERSION)(app)
    app()


if __name__ == '__main__':
    main()

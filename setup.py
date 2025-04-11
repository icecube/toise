#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="toise",
    version="0.1",
    description="Benchmark analysis suite for high-energy, under-ice neutrino detectors",
    author="Jakob van Santen",
    author_email="jakob.van.santen",
    url="https://github.com/icecube/gen2-analysis",
    packages=find_packages(),
    package_data={"toise": ["data/**/*"]},
    entry_points={
        "console_scripts": [
            "toise-figure-data = toise.figures.cli:make_figure_data",
            "toise-plot = toise.figures.cli:make_figure",
            "toise-table = toise.figures.cli:make_table",
        ]
    },
)

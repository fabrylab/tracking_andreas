#!/usr/bin/env python

from setuptools import setup
import os

version='1.0' # adding a version file automatically
file_path=os.path.join(os.getcwd(),os.path.join("pyTrack","_version.py"))
with open(file_path,"w") as f:
	f.write("__version__ = '%s'"%version)

setup(
    name='pyTrack',
    packages=['pyTrack'],
    version=version,
    description='collection of tracking algorithms and database integration',
    url='',
    download_url = '',
    author='Andreas Bauer',
    author_email='andreas.b.bauer@fau.de',
    license='',
    install_requires=[],
    keywords = ['tracking'],
    classifiers = [],
    include_package_data=True,
    )


	

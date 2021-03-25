#!/usr/bin/env python3

from setuptools import setup
from setuptools import find_packages

install_requires = []
dependency_links = []

for package in [l.strip() for l in open('requirements.txt').readlines()]:
    if package.startswith('git+'):
        pk_name = package.split('/')[-1][:-4]
        install_requires.append(f'{pk_name} @ {package}')
    else:
        install_requires.append(package)

setup(
    name='gem_metrics',
    version='0.1dev',
    description='GEM Challenge metrics',
    author='Ondrej Dusek, Aman Madaan, Emiel van Miltenburg, Sebastian Gehrmann, Nishant Subramani, Dhruv Kumar, Miruna Clinciu',
    author_email='odusek@ufal.mff.cuni.cz',
    url='https://github.com/GEM-benchmark/GEM-metrics',
    download_url='https://github.com/GEM-benchmark/GEM-metrics.git',
    license='MIT License',
    install_requires=install_requires,
    dependency_links=dependency_links,
    packages=find_packages(),
    entry_points = {
        'console_scripts': ['gem_metrics=gem_metrics:main']
    }
)

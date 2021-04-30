#!/usr/bin/env python3

import re
from setuptools import setup
from setuptools import find_packages


def load_packages(fname):
    packages = []
    for package in [l.strip() for l in open(fname).readlines()]:
        if package.startswith('git+'):
            if "#egg=" not in package:
                # Intuiting the name by taking last object git url
                # by taking text without any special char "/@." between "/" and ".git":
                # example:
                #   "git+https://github.com/google-research/bleurt.git"
                # returns "bleurt"
                pk_name = re.search(r"(?<=/)([^/@.]+)(?=\.git)", package).group(0)
            else:
                # Taking the given name after keyword "#egg="
                # by taking text without any special char "/@." after "#egg=":
                # example:
                #   "git+https://github.com/recitalAI/QuestEval.git@gem#egg=questeval"
                # returns "questeval"
                pk_name = re.search(r"(?<=#egg=)([^/@.]+)", package).group(0)
            package = f'{pk_name} @ {package}'
        packages.append(package)
    return packages

install_requires = load_packages('requirements.txt')
extras_require = {'heavy': load_packages('requirements-heavy.txt')}


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
    dependency_links=[],
    extras_require=extras_require,
    packages=find_packages(),
    entry_points = {
        'console_scripts': ['gem_metrics=gem_metrics:main']
    }
)

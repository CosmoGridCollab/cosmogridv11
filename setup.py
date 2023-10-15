#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Tomasz Kacprzak",
    author_email='tomasz.kacprzak@psi.ch',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Codes for CosmoGridV1.1 simulations.",
    entry_points={
        'console_scripts': [
            'cosmogridv11=cosmogridv11.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='cosmogridv11',
    name='cosmogridv11',
    packages=find_packages(include=['cosmogridv11', 'cosmogridv11.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tomaszkacprzak/cosmogridv11',
    version='0.1.0',
    zip_safe=False,
)

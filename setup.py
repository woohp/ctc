import os
from setuptools import setup


if 'CC' not in os.environ:
    os.environ['CC'] = 'g++'


setup(
    name='ctc',
    version='0.1',
    description='Connectionist Temporal Classification Loss',
    setup_requires=['cffi>=1.0.0'],
    cffi_modules=['ctc_build.py:ffi'],
    install_requires=['cffi>=1.0.0'],
    packages=['ctc'],
    test_suite='tests'
)

from setuptools import setup, Extension
import tensorflow as tf


setup(
    name='ctc',
    version='0.1',
    description='Connectionist Temporal Classification Loss',
    setup_requires=['cffi>=1.0.0', 'tensorflow'],
    cffi_modules=['ctc_build.py:ffi'],
    install_requires=['cffi>=1.0.0', 'tensorflow'],
    packages=['ctc'],
    ext_modules=[Extension(
        'ctc._tf_op',
        ['src/ctc_tf.cpp', 'src/ctc.cpp'],
        extra_compile_args=['-std=c++1y'],
        include_dirs=[tf.sysconfig.get_include()]
    )],
    test_suite='tests'
)

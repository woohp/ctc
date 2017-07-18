from setuptools import setup, Extension
import sys


extra_compile_args = ['-std=c++1y']


ext_modules = []
try:
    import tensorflow as tf
    ext_modules.append(Extension(
        'ctc._tf_op',
        ['src/ctc_tf.cpp', 'src/ctc.cpp'],
        include_dirs=[tf.sysconfig.get_include()],
        extra_compile_args=extra_compile_args,
    ))
except ImportError:
    pass


setup(
    name='ctc',
    version='0.1',
    description='Connectionist Temporal Classification Loss',
    setup_requires=['cffi>=1.0.0'],
    cffi_modules=['ctc_build.py:ffi'],
    install_requires=['cffi>=1.0.0'],
    packages=['ctc'],
    ext_modules=ext_modules,
    test_suite='tests',
    tests_require=['theano', 'tensorflow>=0.12.1']
)

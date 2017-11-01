from setuptools import setup, Extension
import os


extra_compile_args = ['-std=c++1y']


ext_modules = []
try:
    import tensorflow as tf
    tf_dir = os.path.dirname(tf.__file__)

    ext_modules.append(Extension(
        'ctc._tf_op',
        ['src/ctc_tf.cpp', 'src/ctc.cpp'],
        include_dirs=[tf.sysconfig.get_include()],
        library_dirs=[tf_dir],
        libraries=['tensorflow_framework'],
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
    tests_require=['tensorflow>=1.3.0']
)

from setuptools import setup, find_packages

setup(
    name='dummy_py',
    version='1.7',
    packages=find_packages(),
    url='https://github.com/afpro/dummy_py',
    license='Apache',
    author='afpro',
    author_email='admin@afpro.net',
    description='afpro\'s python library for dummies',
    install_requires=[
        'numpy',
        'librosa',
    ],
    extras_require={
        'tf': ['tensorflow'],
        'tf_gpu': ['tensorflow-gpu'],
    },
    zip_safe=False,
)

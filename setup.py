from setuptools import setup, find_packages

setup(
    name='dummy_py',
    version='1.5',
    packages=find_packages(),
    url='https://github.com/afpro/dummy_py',
    license='Apache',
    author='afpro',
    author_email='admin@afpro.net',
    description='afpro\'s python library for dummies',
    install_requires=[
        'numpy',
    ],
    extras_require={
        "tf": ["tensorflow>=1.4.1"],
        "tf_gpu": ["tensorflow-gpu>=1.4.1"],
    },
    zip_safe=False,
)

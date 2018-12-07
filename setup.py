from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='trainable',
    version='0.1.1.post3',
    description='The flexible training toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jeff Hilton',
    author_email='jeffhilton.code@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='deep-learning dnn training torch',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['torch', 'torchvision', 'tqdm', 'matplotlib', 'numpy']
)

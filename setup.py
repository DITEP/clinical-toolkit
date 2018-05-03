from setuptools import setup, find_packages
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='db-cleansing',

    version='0.1',

    description='Basic preprocessing for clinical databases ',

    long_description=long_description,

    author='Valentin Charvet',

    author_email='valentin.charvet@gustaveroussy.fr',

    packages=find_packages()
    )

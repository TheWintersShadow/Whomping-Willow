# Created by The White Wolf
# Date: 10/31/20
# Time: 2:58 PM
import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="WhompingWillow",
    version="1.0.0",
    author="White Wolf",
    packages=setuptools.find_packages(),
    long_description=long_description,
    install_requires=['pandas', 'gensim', 'pyLDAvis', 'nltk']
)

import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='GCNFrame',
    version='0.0.1',
    packages=find_packages(),
    zip_safe = False,
	url='https://github.com/deepomicslab/GCNFrame',
    license='MIT',
	author='WANG Ruohan',
    author_email='ruohawang2-c@my.cityu.edu.hk',
    description='This is a python package for genomics study with a GCN framework.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy', 'torch==1.7.1', 'torch-geometric==1.7.0', 'biopython==1.78'],
    classifiers=[
		"License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
	include_package_data=True,
    python_requires='>=3.8',
)

import setuptools


# read the contents of the README.md file to get a long_description
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    readme = f.read()
    long_description = readme[:readme.find('## Contents')].strip()

setuptools.setup(
    name="dislib",
    version=open('VERSION').read().strip(),
    author="Barcelona Supercomputing Center",
    author_email="compss@bsc.es",
    description="The distributed computing library on top of PyCOMPSs",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="http://dislib.bsc.es",
    project_urls={
        'Documentation': 'http://dislib.bsc.es',
        'Source': 'https://github.com/bsc-wdc/dislib',
        'Tracker': 'https://github.com/bsc-wdc/dislib/issues',
    },
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    install_requires=[
        "scikit-learn",
        "numpy",
        "scipy",
        "cvxpy"
    ],
    scripts=["bin/dislib", "bin/dislib_cmd.py"],
)

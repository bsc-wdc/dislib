import setuptools
from pkg_resources import parse_requirements as _parse_requirements


def get_long_description():
    """Read the long_description from the README.md file"""
    with open('README.md') as f:
        readme = f.read()
        end = '<!-- End of long_description for setup.py -->'
        return readme[:readme.find(end)].strip()


def parse_requirements():
    """Parse the requirements.txt file"""
    with open('requirements.txt') as f:
        parsed_requirements = _parse_requirements(f)
        requirements = [str(ir) for ir in parsed_requirements]
    return requirements


setuptools.setup(
    name="dislib",
    version=open('VERSION').read().strip(),
    author="Barcelona Supercomputing Center",
    author_email="compss@bsc.es",
    description="The distributed computing library on top of PyCOMPSs",
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url="http://dislib.bsc.es",
    project_urls={
        'Documentation': 'http://dislib.bsc.es',
        'Source': 'https://github.com/bsc-wdc/dislib',
        'Tracker': 'https://github.com/bsc-wdc/dislib/issues',
    },
    packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
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
    install_requires=parse_requirements(),
)

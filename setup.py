import setuptools

setuptools.setup(
    name="dislib",
    version="0.2.1",
    author="Barcelona Supercomputing Center",
    author_email="javier.alvarez@bsc.es",
    description="The distributed computing library on top of PyCOMPSs",
    url="http://dislib.bsc.es",
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
        "scipy"
    ],
)

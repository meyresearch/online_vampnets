import setuptools

setuptools.setup(
    name="celerity",
    version="0.1",
    description="Creates deep MSMs using accelerated training methods",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "addict",
        "click",
        "devtools",
        "mdtraj",
        "numpy",
        "deeptime",
        "h5py",
        "mdshare",
        "pretty-errors",
        "pandas",
        "tables",
        "matplotlib"
    ],

    python_requires=">=3.8",
)
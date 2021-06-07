import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lppydsmc-taltos",
    version="0.0.4",
    author="Paul Calot",
    author_email="paul.calot@hotmail.fr",
    description="small DSMC tool written in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaulCalot/lppydsmc",
    project_urls={
        "Bug Tracker": "https://github.com/PaulCalot/lppydsmc/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)


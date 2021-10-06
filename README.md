# LPPy-DSMC


## Version

- V0.0.1 - 2021/10/06: first addition of the version counter. Proceeding the correction of a bug that prevented the use of custom *reflect_fn* function and the correction of the interpolation points for the background gas so they correspond to the middle of each cell. Small changes in the documentation too.

## Introduction

**LPPy-DSMC** is both a library containing useful procedures and a script meant to be run using configuration files. It uses `Python` programming language.

It was designed as a `Direct Simulation Monte Carlo` tool, initially to simulate a rarefied gas flow throught a somewhat complex geometry. It was then improved until its current state so it can offer a more comprehensive approach. The [Documentation](#documentation) introduced more thoroughly its functionnalities and limits.

## Table of contents
- [LPPy-DSMC](#lppy-dsmc)
  - [Version](#version)
  - [Introduction](#introduction)
  - [Table of contents](#table-of-contents)
  - [Setting the environment](#setting-the-environment)
    - [Basic](#basic)
    - [Basic + plotting](#basic--plotting)
    - [Basic + fenics](#basic--fenics)
    - [Basic + fenics + plotting](#basic--fenics--plotting)
  - [Installing `lppydsmc`](#installing-lppydsmc)
  - [Running a simulation - config files](#running-a-simulation---config-files)
    - [1. Configuration of the simulation](#1-configuration-of-the-simulation)
    - [2. Launching a simulation](#2-launching-a-simulation)
    - [3. Analysis of the results](#3-analysis-of-the-results)
  - [Specifications file](#specifications-file)
  - [Documentation](#documentation)
    - [Introduction](#introduction-1)
    - [Functionalities](#functionalities)


## Setting the environment

Several versions of the environment is available, depending on weather you want to use only the basic tools, the poisson solver with [fenics](https://fenicsproject.org/) or the plotting tools too.

### Basic 
```
conda create --name NAME_ENV python=3.7 pandas numpy tqdm numexpr pytables configobj
```
### Basic + plotting
```
conda create --name NAME_ENV python=3.7 pandas numpy tqdm numexpr pytables configobj matplotlib seaborn 
```
### Basic + fenics
```
conda create --name NAME_ENV -c conda-forge fenics=2018 mshr=2018 pandas numpy tqdm numexpr pytables configobj
```

Note : *On 25th July 2021, the 2019 install using conda, for *fenics* and *mshr*, did not work properly.* The 2018 versions forces the use of `python 3.7` which is the reasons why it is forced also on the other build versions. If you install the 2019 version in another way, the code should still work for the *fenics* part, but there is no guarantee no parts of it will break.

### Basic + fenics + plotting
```
conda create --name NAME_ENV -c conda-forge fenics=2018 mshr=2018 pandas numpy tqdm numexpr pytables configobj seaborn matplotlib
```

Please add to the command :
- `notebook` if you want to have *jupyter notebook* available,
- `jupyterlab` if you want to have *jupyter lab* available

## Installing `lppydsmc`

The best way to make a package available and usable from a notebook or terminal and from any directory is to install it in the environment, at least locally.

To install `lppydsmc`:
```shell
git clone https://calot@hephaistos.lpp.polytechnique.fr/rhodecode/GIT_REPOSITORIES/LPP/Users/Calot/lppydsmc
cd lppydsmc
git checkout main
conda activate NAME_ENV
python -m pip install e .
```
At this point, the package *lppydsmc* will be available in the environment NAME_ENV and you can import it like you would do with `NumPy` and from anywhere:
```python
import lppydsmc as ld
```

Since it was installed using `pip`, you can uninstall it simply by doing `pip uninstall lppydsmc-taltos` in a terminal.

> Note: I plan on releasing it to `Pypi` as a package that could be installed using `pip` directly.

## Running a simulation - config files

Using the configuration files and the very high level `lppydsmc.main`.

### 1. Configuration of the simulation 
Please, see the [specifications](lppydsmc/config/spec.ini) files or the [examples](./benchmarks/).

Available options :
- Saving name and directory
- Simulation parameters
- System
- Species
- Particles initialization (optional)
- Injection (optional)
- DSMC (optional)
- background gas (optional)
- Reactions (optional)
- Poisson solver (optional, requires *fenics* install)
- Monitoring (optional)
- Verbose (optional)

### 2. Launching a simulation
- Command line (you need to be in the folder where `run.py` is):
   ```shell
   conda activate ENV_NAME
   python run.py -p <path_to_cfg_file> -s 
   ```
The `-s` flag simply tells the algorithm to save the parameters at the end of the setup phase, right before the simulation begins.

- Jupyter notebook / lab with the right environment:
    ```python
    import lppydsmc as ld
    path = <path_to_cfg_file>
    ld.main(path, save = True) # save is True by default
    ```

At this point, you should see a progress bar with the current iteration and estimated remaining time before completion (depending on what you chose in the options - this is the default case).

### 3. Analysis of the results
If you chose to monitor the simulations in your options in the configuration file, then an `hdf5` will be available under the path `<dir_path>/<name>/monitoring.h5` where *directory_path* and *name* comes from the configuration file.

You can load the `hdf5` file of your simulation this way :
```python
import pandas as pd
store = pd.HDFStore(results_path)
print(store.keys()) # to see what has been saved (depends on what you activated amongst the optional options)
```
You can then explore each dataframes.

Depending on the options you chose, you will find a directory `images` with plots of the state of the system and of the velocity distribution and a file `params.ini` that summarizes the parameters used in the simulation. It encompasses the computed values from the "raw" configuration file.

## Specifications file
The module takes advantage of the [ConfigObj](https://github.com/DiffSK/configobj) module. A good tutorial is available [here](http://www.voidspace.org.uk/python/articles/configobj.shtml) and you can find the documentation [here](https://configobj.readthedocs.io/en/latest/).

You can find the specification file [here](lppydsmc/config/spec.ini). 

> Note : it will be heavily commented in the future, once the code is under less development.

## Documentation 
**(IN PROGRESS)**

The current state of the project, the available functionalities and limitations along with other useful information are presented here.

### Introduction

This code was initially designed to simulate an flow of a rarefied gas made of neutrals atoms through a somewhat complex geometry.

It was then designed to be a closer to a generic tool but design choices were made in agreement with the final goal aforementioned.

### Functionalities 

The code was from the start designed with flexibility in mind. For that reason, lots of options are available which makes the configuration files slightly harder to digest.

Also note that while most functionnalities were unit tested, especially the more important ones, the code has not passed enough integration benchmarks, nor mezzanine ones, to be considered with any degree of certainty correct.

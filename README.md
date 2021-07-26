# LPPy-DSMC

## Installing the environment


Several versions of the environment is available, depending on weather you want to use only the basic tools, the poisson solver with [fenics](https://fenicsproject.org/) or the plotting tools too.

TODO : scipy ?

### Basic 
```
conda create --name lpp python=3.7 pandas numpy tqdm numexpr pytables configobj
```
### Basic + plotting
```
conda create --name lpp python=3.7 pandas numpy tqdm numexpr pytables configobj matplotlib seaborn 
```
### Basic + fenics
```
conda create --name lpp -c conda-forge fenics=2018 mshr=2018 pandas numpy tqdm numexpr pytables configobj
```

Note : *On 25th July 2021, the 2019 install using conda, for *fenics* and *mshr*, did not work properly.* The 2018 versions forces the use of `python 3.7` which is the reasons why it is forced also on the other build versions. If you install the 2019 version in another way, the code should still work for the *fenics* part, but there is no guarantee no parts of it will break.

### Basic + fenics + plotting
```
conda create --name lpp -c conda-forge fenics=2018 mshr=2018 pandas numpy tqdm numexpr pytables configobj seaborn matplotlib
```

Please add to the command :
- `notebook` if you want to have *jupyter notebook* available,
- `jupyterlab` if you want to have *jupyter lab* available


### Running the notebooks (NOT WORKING)
To run the notebooks, you will need [ipympl](https://github.com/matplotlib/ipympl) which allows interacting with `matplotlib` plots inside *jupyter lab*. If you do not want to install it, simply replace `%matplotlib widget` by either `%matplotlib notebook` for *jupyter notebook* or `%matplotlib inline` for *jupyter lab*.

## Running a simulation

Coming soon (if I have the time): 
> You can use the `main.ipynb` notebook to launch easily a first simulation and then you can analyze it using the `analysis.ipynb` notebook.

Using the configuration files and the very high level `lppydsmc.run`:
1. Configuration of the simulation (cf. the [specifications](lppydsmc/config/spec.ini) files or the [examples](lppydsmc/config/spec.ini)).
   1. Saving name and directory
   2. Simulation parameters
   3. System
   4. Species
   5. Particles initialization (optional)
   6. Injection (optional)
   7. DSMC (optional)
   8. Reactions (optional)
   9. Poisson solver (optional, requires *fenics* install)
   10. Monitoring (optional)
   11. COMING SOON : Plotting and verbose (optional, requires *plotting* install)
2. Launch the simulation:
   1. Command line:
    ```shell
    conda activate ENV_NAME
    python main -p <path_to_cfg_file>
    ```
   2. Jupyter notebook / lab with the right environment:
    ```python
    import lppydsmc as ld
    path = <path_to_cfg_file>
    ld.run(path, save = True) # save is True by default
    ```
    At this point, you should see a progress bar with the current iteration and estimated remaining time before completion.
3. Once the simulation has finished. You can load the `hdf5` file of your simulation :
   ```python
   import pandas as pd
   store = pd.HDFStore(results_path)
   print(store.keys()) # to see what has been saved (depends on what you activated amongst the optional options)
   ```
   You can then explore each dataframes.

A set a useful plotting tools is available under the folder [plotting](plotting/) [TO BE IMPROVED].

## Specifications file
The module takes advantage of the [ConfigObj](https://github.com/DiffSK/configobj) module. A good tutorial is available [here](http://www.voidspace.org.uk/python/articles/configobj.shtml) and you can find the documentation [here](https://configobj.readthedocs.io/en/latest/).

COMING SOON : the specifications file, heavily commented, will be available directly in the README. For now, you can find it [here](lppydsmc/config/spec.ini).

## Documentation (IN PROGRESS)

The current state of the project, the available functionalities and limitations along with other useful information are presented here.

### Introduction

This code was initially designed to simulate an flow of a rarefied gas made of neutrals atoms through a somewhat complex geometry.

It was then designed to be a closer to a generic tool but design choices were made in agreement with the final goal aforementioned.

### Functionalities 

The code was from the start designed with flexibility in mind. For that reason, lots of options are available which makes the configuration files slightly harder to digest.

Also note that while most functionnalities were unit tested, especially the more important ones, the code has not passed enough integration benchmarks, nor mezzanine ones, to be considered with any degree of certainty correct.

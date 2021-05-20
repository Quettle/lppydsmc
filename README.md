# LPPy-DSMC

## Launch unit test first
**NOT UP TO DATE - won't work**
```shell
python -m unittest discover
```

## Installing the environment
```
conda create --name lpp python=3.9 pandas numpy matplotlib tqdm notebook jupyterlab numexpr pytables seaborn
```

To run the notebooks, you will need [ipympl](https://github.com/matplotlib/ipympl) which allows interacting with `matplotlib` plots inside *jupyter lab*. If you do not want to install it, simply replace `%matplotlib widget` by either `%matplotlib notebook` for *jupyter notebook* or `%matplotlib inline` for *jupyter lab*.

## Running a simulation

You can use the `main.ipynb` notebook to launch easily a first simulation and then you can analyze it using the `analysis.ipynb` notebook.
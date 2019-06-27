# Polymetalic nodules occurrence model

## Installing dependencies
To run the code it is recommended to setup a local Python3.7 conda environment
```sh
$ conda create -n polymetallic-nodules python=3.7 numpy scipy pandas xlrd scikit-learn netcdf4 xarray jupyter matplotlib
$ source activate polymetallic-nodules
```
The project-specific utilities may then be installed
```sh
$ pip install -e utils
```

## Running the code
All data files are required to be present in the `data` directory.
Running the notebook will produce the following materials:
* Nearest neighbour interpolated variable grids (`grids.nc`)
* Nodule and control point data files (`nodules.csv`, `control.csv`)
* Grid data interpolated on a Fibonnacci lattice (`lattice.csv`)
* Kolmogorov-Smirnov statistics for comparison of oceanic variable samples at exploration and lattice points (`ks_stats.csv`)
* Mutual information estimate between variable and nodule occurrence probability grids (`mi.csv`)
* Mutual information bar graph (`mi.pdf`)
* Variable dependence plots (`variable_dependence.pdf`)
* Nodule occurrence probability grids (`probability_grids.nc`)

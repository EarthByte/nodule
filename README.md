# Polymetalic nodules occurrence model

## Reference
This github repository is a supplement to the paper: Dutkiewicz, A., Judge, A. and Müller, R.D., 2020, Environmental predictors of deep-sea polymetallic nodule occurrence in the global ocean, Geology, 48, p. XXX–XXX, https://doi.org/10.1130/G46836.1

## Installing dependencies
To run the code it is recommended to setup a local Python3.7 conda environment on a macOS or Ubuntu Linux machine
```sh
$ conda create -n polymetallic-nodules python=3.7 numpy scipy pandas xlrd scikit-learn netcdf4 xarray jupyter matplotlib
$ conda activate polymetallic-nodules
```
The project-specific utilities may then be installed
```sh
$ pip install -e utils
```

## Running the code
All data files are required to be present in the `data` directory, with the oceanic variable grids under `data/grids` and the files containing the nodule and control lat-lon points under `data/csv`. Make sure that you have installed "LaTex".
Running the notebook will produce the following materials:
* Nearest neighbour interpolated variable grids (`grids.nc`)
* Nodule and control point data files with variable values interpolated at exploration points (`nodules.csv`, `control.csv`)
* Grid data interpolated on a Fibonnacci lattice (`lattice.csv`)
* Kolmogorov-Smirnov statistics for comparison of oceanic variable samples at exploration and lattice points (`ks_stats.csv`)
* Mutual information estimate between variable and nodule occurrence probability grids (`mi.csv`)
* Mutual information bar graph (`mi.pdf`)
* Variable dependence plots (`variable_dependence.pdf`)
* Nodule occurrence probability grids (`probability_grids.nc`)

Expected runtime for the full notebook on a standard laptop is approximately 5hrs. Most of the computation is concerned with estimating the mutual information between the nodule occurrence probability and the variable grids.

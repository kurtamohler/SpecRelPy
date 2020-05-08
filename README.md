# SpecRelPy
Library for special relativity physics

## Purpose
I've been learning about special relativity, and I wanted to test my knowledge by writing a library to help with special relativity calculations or simulations.

Right now, there's not much here, but here's what's here:

* [lorentz_transform.py](lorentz_transform.py) - The basic space-time Lorentz transform with support for N spatial dimensions.
* [lorentz_transform_visualization.ipynb](lorentz_transform_visualization.ipynb) - An interactive visualization of a Lorentz transformation of a 2-D square.

## How to run

### Using conda

* Create and activate a conda environment
```
$ conda env create -n specrelpy -f environment.yaml
$ conda activate specrelpy
```

* Run the project with jupyter
```
$ jupyter notebook
```

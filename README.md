# Project on supercurrents in nanowire Josephson junctions
For the paper: Supercurrent interference in few-mode nanowire Josephson junctions 
by Vincent Mourik, Daniel B. Szombati, Bas Nijholt, David J. van Woerkom,
Attila Geresdi, Jun Chen, Viacheslav P. Ostroukh, Anton R. Akhmerov,
Sebastian R. Plissard, Diana Car, Erik P. A. M. Bakkers, Dmitry I. Pikulin,
Leo P. Kouwenhoven, and Sergey M. Frolov.

Code written by Bas Nijholt.


# Files
This folder contains three Jupyter notebooks and three Python files:
* `generate-data.ipynb`
* `explore-data.ipynb`
* `paper-figures.ipynb`
* `funcs.py`
* `common.py`
* `combine.py`

Most of the functions used in `generate-data.ipynb` are defined in `funcs.py`.

All notebooks contain instructions of how it can be used.

## generate-data.ipynb
Generates numerical data used in the paper.

## explore-data.ipynb
Interactively explore data files uploaded on the 4TU library. See for example
current-phase relations for different system lengths, disorder strengths, with
or without the spin-orbit or Zeeman effect, different temperatures, and more!

## paper-figures.ipynb
Plot the figures that are found in the paper.


# Data
Download the data used in `explore-data.ipynb` and `paper-figures.ipynb` at http://doi.org/10.4121/uuid:274bdd06-14a5-45c3-bc86-87d400082e34


# Installation
Install [miniconda](http://conda.pydata.org/miniconda.html) and then the Python 
environment that contains all dependencies with:

```
conda env create -f environment.yml -n kwant
```

Run `jupyter-notebook` to open the `*.ipynb` files.
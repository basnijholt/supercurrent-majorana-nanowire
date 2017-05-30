# Project on supercurrents in nanowire Josephson junctions
For the paper: "Supercurrent interference in few-mode nanowire Josephson junctions"
by Vincent Mourik, Daniel B. Szombati, Bas Nijholt, David J. van Woerkom,
Attila Geresdi, Jun Chen, Viacheslav P. Ostroukh, Anton R. Akhmerov,
Sebastian R. Plissard, Diana Car, Erik P. A. M. Bakkers, Dmitry I. Pikulin,
Leo P. Kouwenhoven, and Sergey M. Frolov.

Code written by Bas Nijholt.


# Files
This folder contains five Jupyter notebooks and three Python files:
* `generate-data.ipynb`
* `explore-data.ipynb`
* `mean-free-path.ipynb`
* `paper-figures.ipynb`
* `example-toy-models.ipynb`
* `funcs.py`
* `common.py`
* `combine.py`

Most of the functions used in `generate-data.ipynb` are defined in `funcs.py`.

All notebooks contain instructions of how it can be used.

## generate-data.ipynb
Generates numerical data used in the paper.

## mean-free-path.ipynb
Calculates the mean-free path using the data that is generated in `generate-data.ipynb`.

## explore-data.ipynb
Interactively explore data files uploaded on the 4TU library. See for example
current-phase relations for different system lengths, disorder strengths, with
or without the spin-orbit or Zeeman effect, different temperatures, and more!

## paper-figures.ipynb
Plot the figures that are found in the paper.

## simple-example-toy-models.ipynb
Contains simple toy models and examples of how to calculate the current-phase relations.


# Data
Download the data used in `explore-data.ipynb` and `paper-figures.ipynb` at UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU_UPLOAD_DATA_TO_4TU and put in in this folder as `data/`.


# Installation
Install [miniconda](https://conda.io/miniconda.html) and add a Python 
environment that contains all dependencies with:

```
conda env create -f environment.yml -n kwant
```

Run `jupyter-notebook` in your terminal to open the `*.ipynb` files.

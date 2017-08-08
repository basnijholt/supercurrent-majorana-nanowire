# Project on supercurrents in nanowire Josephson junctions
For the paper: "Supercurrent interference in few-mode nanowire Josephson junctions"
by Kun Zuo, Vincent Mourik, Daniel B. Szombati, Bas Nijholt, David J. van Woerkom,
Attila Geresdi, Jun Chen, Viacheslav P. Ostroukh, Anton R. Akhmerov,
Sebastian R. Plissard, Diana Car, Erik P. A. M. Bakkers, Dmitry I. Pikulin,
Leo P. Kouwenhoven, and Sergey M. Frolov.

Code written by Bas Nijholt, Viacheslav P. Ostroukh, Anton R. Akhmerov, and Dmitry I. Pikulin.


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

## [`generate-data.ipynb`](generate-data.ipynb)
Generates numerical data used in the paper.

## [`mean-free-path.ipynb`](mean-free-path.ipynb)
Estimates the mean-free path using the data that is generated in `generate-data.ipynb`.

## [`explore-data.ipynb`](explore-data.ipynb)
Interactively explore data files uploaded on the 4TU library. See for example
current-phase relations for different system lengths, disorder strengths, with
or without the spin-orbit or Zeeman effect, different temperatures, and more!

## [`paper-figures.ipynb`](paper-figures.ipynb)
Plot the figures that are found in the paper.

## [`simple-example-toy-models.ipynb`](simple-example-toy-models.ipynb)
Contains simple toy models and examples of how to calculate the current-phase relations.


# Data
Download the data used in `explore-data.ipynb` and `paper-figures.ipynb` at http://doi.org/10.4121/uuid:daa83d96-85d6-4739-841e-52abc86b6b15 and put in in this folder as `data/`.


# Installation (Windows/OS X/Linux)

* (Windows only) Install [Windows Subsystem for Linux (WSL)](https://msdn.microsoft.com/en-us/commandline/wsl/install_guide).
* (Windows only) Open "Bash on Ubuntu on Windows" and choose the Linux version of Miniconda in the next steps!

* Download [miniconda](https://conda.io/miniconda.html) for Python 3 and [install](https://conda.io/docs/install/quick.html) it.

* (Windows only) Run the following command
```
echo 'export PATH="${HOME}/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

* Enter the directory with these downloaded files, with `cd`, for example:
```
cd ~/Downloads/supercurrent_data
```

* Then add a Python environment that contains all dependencies with:
```
conda env create -f environment.yml -n supercurrent
```

* To activate the environment that we just installed run:
```
source activate supercurrent
```

* Run `jupyter-notebook` in your terminal to open the `*.ipynb` files.


# Alternative installation (only for the `explore-data.ipynb` notebook) (Windows/OS X/Linux)
If you do not have a working Python 3.6 environment:
* Download [miniconda](https://conda.io/miniconda.html) for Python 3.6 and [install](https://conda.io/docs/install/quick.html) it.

If you already have `conda` do or just installed it, run:
```
conda env create -c conda-forge -n supercurrent python=3.6 holoviews=1.8 pandas pytables toolz numpy notebook
source activate supercurrent
jupyter notebook
```

If you already have a Python 3.6 environment without `conda`, use `pip`:
```
pip install -U holoviews==1.8.1 pandas tables toolz numpy notebook
jupyter notebook
```
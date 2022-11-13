# Project on supercurrents in nanowire Josephson junctions
For the paper: "Supercurrent interference in few-mode nanowire Josephson junctions"
by Kun Zuo, Vincent Mourik, Daniel B. Szombati, Bas Nijholt, David J. van Woerkom,
Attila Geresdi, Jun Chen, Viacheslav P. Ostroukh, Anton R. Akhmerov,
Sebastian R. Plissard, Diana Car, Erik P. A. M. Bakkers, Dmitry I. Pikulin,
Leo P. Kouwenhoven, and Sergey M. Frolov.

Code written by Bas Nijholt, Viacheslav P. Ostroukh, Anton R. Akhmerov, and Dmitry I. Pikulin.

The `data` folder contains the following files:
* `data/1d_alpha_vs_B_x_T0.1K.hdf`
* `data/1d_alpha_vs_B_x_T1.0K.hdf`
* `data/I_c(B_x)_mu10,20meV_disorder0,75meV_T0.1K_all_combinations_of_effects.hdf`
* `data/I_c(B_x)_mu20meV_rotation_of_field_in_xy_plane.hdf`
* `data/I_c(B_x)_no_disorder_combinations_of_effects_and_geometries.hdf`
* `data/I_c(B_x,_V)_gate160nm_mu20meV_disorder0,75meV_T0.1K.hdf`
* `data/gap_tuning.hdf`
* `data/mean_free_path.hdf`
* `data/experimental_data/fig1.mtx`
* `data/experimental_data/fig2a.mtx`
* `data/experimental_data/fig2b.mtx`
* `data/experimental_data/fig2c.mtx`
* `data/experimental_data/fig2d.mtx`
* `data/experimental_data/fig3a.mtx`
* `data/experimental_data/fig3b.mtx`
* `data/experimental_data/fig3c.mtx`
* `data/experimental_data/fig3d.mtx`
* `data/experimental_data/fig3e.mtx`
* `data/experimental_data/fig5.txt`
* `data/experimental_data/sup_fig1c.mtx`
* `data/experimental_data/sup_fig2a.mtx`
* `data/experimental_data/sup_fig2c.mtx`
* `data/experimental_data/sup_fig2d.mtx`
* `data/experimental_data/sup_fig2e.mtx`
* `data/experimental_data/sup_fig2f.mtx`
* `data/experimental_data/sup_fig2g.mtx`
* `data/experimental_data/sup_fig3a.mtx`
* `data/experimental_data/sup_fig3b.mtx`
* `data/experimental_data/sup_fig3c.mtx`
* `data/experimental_data/sup_fig5a.mtx`
* `data/experimental_data/sup_fig5b.mtx`
* `data/experimental_data/sup_fig5d.mtx`

The `*.hdf` files are HDF5 [`pandas`](http://pandas.pydata.org/) `DataFrame` objects.
The `*.mtx` files are [SpyView](http://nsweb.tn.tudelft.nl/~gsteele/spyview/) files and can be opened with a function `open_mtx` defined in `paper_figures.ipynb`.

See the `CODE_README.md` for instructions on how to use the code.

The `explore-data.ipynb` and `paper-figures.ipynb` notebook can be used to visualize the data.

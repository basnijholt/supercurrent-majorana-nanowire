#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017, Bas Nijholt
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the TU Delft nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL BAS NIJHOLT BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# General functions used for saving, processing, and generating data.

from datetime import timedelta
import functools
from glob import glob
import inspect
from itertools import product
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from toolz import partition_all

assert sys.version_info >= (3, 6), 'Use Python ≥3.6'


def upd(d, **kwargs):
    # Update a `dict` inline and return the `dict`.
    d = d.copy()
    for k, v in kwargs.items():
        d[k] = v
    return d


def run_simulation(lview, func, vals, parameters, fname_i, N=None,
                   overwrite=False):
    """Run a simulation where one loops over `vals`. The simulation
    yields len(vals) results, but by using `N`, you can split it up
    in parts of length N.

    Parameters
    ----------
    lview : ipyparallel.client.view.LoadBalancedView object
        LoadBalancedView for asynchronous map.
    func : function
        Function that takes a list of arguments: `vals`.
    vals : list
        Arguments for `func`.
    parameters : dict
        Dictionary that is saved with the data, used for constant
        parameters.
    fname_i : str
        Name for the resulting HDF5 files. If the simulation is
        split up in parts by using the `N` argument, it needs to
        be a formatteble string, for example 'file_{}'.
    N : int
        Number of results in each pandas.DataFrame.
    overwrite : bool
        Overwrite the file even if it already exists.
    """
    if N is None:
        N = 1000000
        if len(vals) > N:
            raise Exception('You need to split up vals in smaller parts')

    N_files = len(vals) // N + (0 if len(vals) % N == 0 else 1)
    print('`vals` will be split in {} files.'.format(N_files))
    time_elapsed = 0
    parts_done = 0
    for i, chunk in enumerate(partition_all(N, vals)):
        fname = fname_i.replace('{}', '{:03d}').format(i)
        print('Busy with file: {}.'.format(fname))
        if not os.path.exists(fname) or overwrite:
            map_async = lview.map_async(func, chunk)
            map_async.wait_interactive()
            result = map_async.result()
            df = pd.DataFrame(result)
            df = df.assign(**parameters)
            df = df.assign(git_hash=get_git_revision_hash())
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            df.to_hdf(fname, 'all_data', mode='w', complib='zlib', complevel=9)

            # Print useful information
            N_files_left = N_files - (i + 1)
            parts_done += 1
            time_elapsed += map_async.elapsed
            time_left = timedelta(seconds=(time_elapsed / parts_done) *
                                  N_files_left)
            print_str = ('Saved {}, {} more files to go, {} time left '
                         'before everything is done.')
            print(print_str.format(fname, N_files_left, time_left))
        else:
            print('File: {} was already done.'.format(fname))


def change_var_name(func, from_name, to_name):
    sig = inspect.signature(func)
    pars = [(name, value) for name, value in sig.parameters.items()]

    new_pars = []
    for k, v in pars:
        if k is not from_name:
            new_pars.append(v)
        else:
            new_pars.append(inspect.Parameter(to_name, v.kind,
                                              default=v.default))

    def wrapped(*args, **kwargs):
        kwargs[from_name] = kwargs.pop(to_name)
        return func(*args, **kwargs)

    wrapped.__signature__ = inspect.Signature(parameters=new_pars)

    return wrapped


def parse_params(params):
    for k, v in params.items():
        if isinstance(v, str):
            try:
                params[k] = eval(v)
            except NameError:
                pass
    return params


def combine_dfs(pattern, fname=None):
    files = glob(pattern)
    df = pd.concat([pd.read_hdf(f) for f in sorted(files)])
    df = df.reset_index(drop=True)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        df.to_hdf(fname, 'all_data', mode='w', complib='zlib', complevel=9)

    return df


def lat_from_syst(syst):
    lats = set(s.family for s in syst.sites)
    if len(lats) > 1:
        raise Exception('No unique lattice in the system.')
    return list(lats)[0]


def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer


def named_product(**items):
    names = items.keys()
    vals = items.values()
    return [dict(zip(names, res)) for res in product(*vals)]


def get_git_revision_hash():
    """Get the git hash to save with data to ensure reproducibility."""
    git_output = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return git_output.decode("utf-8").replace('\n', '')


def find_nearest(array, value):
    """Find the nearest value in an array to a specified `value`."""
    idx = np.abs(np.array(array) - value).argmin()
    return array[idx]


def remove_unhashable_columns(df):
    df = df.copy()
    for col in df.columns:
        if not hashable(df[col].iloc[0]):
            df.drop(col, axis=1, inplace=True)
    return df


def hashable(v):
    """Determine whether `v` can be hashed."""
    try:
        hash(v)
    except TypeError:
        return False
    return True


def drop_constant_columns(df):
    """Taken from http://stackoverflow.com/a/20210048/3447047"""
    df = remove_unhashable_columns(df)
    df = df.reset_index(drop=True)
    return df.loc[:, (df != df.ix[0]).any()]

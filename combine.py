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
# Defines a function `combine` that can create a new function by combining two
# functions. For example `combine(f, g, operator.mul)` creates a function
# that multiplies `f` with `g`.

import inspect
from collections import OrderedDict

def get_names(sig):
    names = [(name, value) for name, value in sig.parameters.items() if
             value.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY)]
    return OrderedDict(names)


def filter_kwargs(sig, names, kwargs):
    names_in_kwargs = [(name, value) for name, value in kwargs.items()
                       if name in names]
    return OrderedDict(names_in_kwargs)


def skip_pars(names1, names2, num_skipped):
    skipped_pars1 = list(names1.keys())[:num_skipped]
    skipped_pars2 = list(names2.keys())[:num_skipped]
    if skipped_pars1 == skipped_pars2:
        pars1 = list(names1.values())[num_skipped:]
        pars2 = list(names2.values())[num_skipped:]
    else:
        raise Exception('First {} arguments '
                        'have to be the same'.format(num_skipped))
    return pars1, pars2


def combine(f, g, operator, num_skipped=0):
    if not callable(f) or not callable(g):
        raise Exception('One of the functions is not a function')

    sig1 = inspect.signature(f)
    sig2 = inspect.signature(g)

    names1 = get_names(sig1)
    names2 = get_names(sig2)

    pars1, pars2 = skip_pars(names1, names2, num_skipped)
    skipped_pars = list(names1.values())[:num_skipped]

    def wrapped(*args, **kwargs):
        kwargs1 = filter_kwargs(sig1, names1.keys(), kwargs)
        kwargs2 = filter_kwargs(sig2, names2.keys(), kwargs)

        fval = f(*args, **kwargs1)
        gval = g(*args, **kwargs2)
        return operator(fval, gval)

    pars1_names = [p.name for p in pars1]
    pars2 = [p for p in pars2 if p.name not in pars1_names]
    parameters = pars1 + pars2
    parameters = [p.replace(kind=inspect.Parameter.KEYWORD_ONLY,
                            default=p.default) for p in parameters]
    wrapped.__signature__ = inspect.Signature(parameters=skipped_pars + parameters)
    return wrapped

#! /usr/bin/env python

"""Preprocess raw trace data, so it can be used for standard analysis downstream.

author: krissankaran@stanford.edu
date: 03/21/2018
"""

import numpy as np
import pandas as pd
import os


def process_names(fnames):
    """Extract Metadata from Filenames

    The files are named in a consistent enough way that we can extract the
    relevant features automatically.

    :param fnames: The original names as a list of strings, encoding features
        for each sample.
    :return metadata: A pd.DataFrame whose rows are samples and whose columns.

    >>> raw_path = os.path.join("data", "raw", "raw data")
    >>> fnames = os.listdir(raw_path)
    >>> process_names(fnames)
    """
    fnames = [x.replace("msl 5", "msl5") for x in fnames] # seems like an error with naming
    split_names = [os.path.splitext(x)[0].split(" ") for x in fnames]
    metadata = pd.DataFrame(split_names, columns = ["genotype", "ch1", "ch2", "ch3", "ch4"])
    metadata.insert(0, "fname", fnames)
    for i in range(1, 5):
        cur_ix = "ch" + str(i)
        metadata[cur_ix] = metadata[cur_ix].map(lambda x: x.lstrip(cur_ix))

    return metadata


def read_traces(fnames):
    """Read Traces from File

    :param fnames: The original names as a list of strings, encoding features
        for each sample.
    :return traces: A pandas DataFrame containing the raw data.

    >>> raw_path = os.path.join("data", "raw", "raw data")
    >>> fnames = os.listdir(raw_path)
    >>> traces = read_traces([os.path.join(raw_path, f) for f in fnames])
    >>>
    >>> from plotnine import *
    >>> (ggplot(traces[traces.time < 550].iloc[::250, :]) +
            geom_line(aes(x = "time", y = "value", color = "channel")) +
            geom_vline(
                aes(xintercept = "time"),
                data = traces[traces.cut_point == "cut"],
            ) +
            ylim(-0.1, 0.07) +
            facet_wrap("fname", scales = "fixed") +
            theme(
                strip_text = element_blank(),
                axis_text = element_blank(),
                axis_ticks = element_blank()
            )
        )
    """
    traces = dict()
    for fname in fnames:

        print("Processing " + fname)
        cur_trace = []
        with open(fname, "r") as f:
            f.readline() # skip header

            for line in f:
                cur_vals = [fname]
                cur_vals += line.rstrip("\n").split("\t")
                if len(cur_vals) < 7:
                    cur_vals += ["no_cut"]

                cur_trace.append(cur_vals)

        cols = ["fname", "time", "ch1", "ch2", "ch3", "ch4", "cut_point"]
        traces[fname] = pd.DataFrame(cur_trace, columns=cols)
        float_ix = ["time"] + ["ch" + str(x) for x in range(1, 5)]
        traces[fname][float_ix] = traces[fname][float_ix].astype(np.float64)

    return pd.melt(
        pd.concat(traces, ignore_index = True),
        id_vars = ["fname", "time", "cut_point"],
        var_name = "channel"
    )

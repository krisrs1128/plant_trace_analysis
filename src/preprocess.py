#! /usr/bin/env python

"""


Preprocess raw trace data, so it can be used for standard analysis downstream.

author: krissankaran@stanford.edu
date: 03/21/2018
"""

import os
import pandas as pd


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


def combine_traces(x):
    """Combine Raw Traces into Data Array

    :param x: A dictionary of DataFrames, each of which is a raw data trace for
        a single sample.
    :return df: A numpy array containing data across all the samples.
    """
    pass


def read_traces(fnames):
    """Read Traces from File

    :param fnames: The original names as a list of strings, encoding features
        for each sample.
    :return traces: A dictionary of DataFrames containing the raw data.

    >>> raw_path = os.path.join("data", "raw", "raw data")
    >>> fnames = os.listdir(raw_path)
    >>> traces = read_traces([os.path.join(raw_path, f) for f in fnames])
    """
    traces = dict()
    for fname in fnames:

        print("Processing " + fname)
        cur_trace = []
        with open(fname, "r") as f:
            f.readline() # skip first line

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

    return pd.concat(traces, ignore_index=True)



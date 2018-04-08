#! /usr/bin/env python

"""Preprocess raw trace data, so it can be used for standard analysis downstream.

author: krissankaran@stanford.edu
date: 03/21/2018
"""

import numpy as np
import pandas as pd
import os
import re


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
    split_names = [os.path.splitext(x)[0].split(" ") for x in fnames]
    metadata = pd.DataFrame(split_names, columns = ["genotype_leaf", "ch1", "ch2", "ch3", "ch4"])
    metadata.insert(0, "fname", fnames)
    for i in range(1, 5):
        cur_ix = "ch" + str(i)
        metadata[cur_ix] = metadata[cur_ix].map(lambda x: x.lstrip(cur_ix))

    genotype = [re.search("[A-z]+", s[0]).group() for s in split_names]
    source_leaf = ["L" + re.search("[0-9]+", s[0]).group() for s in split_names]
    metadata.insert(1, "source", source_leaf)
    metadata.insert(1, "genotype", genotype)
    metadata = metadata.drop("genotype_leaf", 1)

    return pd.melt(
        metadata,
        id_vars=["fname", "genotype", "source"],
        var_name="channel",
        value_name="target"
    )


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
    >>> (ggplot(traces) +
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
        id_vars = ["fname", "cut_point", "time"],
        var_name = "channel"
    )


def align_trace(trace):
    cut_time = trace.loc[trace["cut_point"] == "cut", "value"]
    trace["time"] = trace["time"] - float(cut_time)
    return trace


def merge_meta(metadata, combined, cur_times, paste_times):
    lpaste = len(paste_times)
    paste_df = pd.concat([
        metadata.iloc[np.zeros(lpaste), :].reset_index(drop=True),
        pd.DataFrame(
            {"cut_point": np.repeat("no_cut", lpaste),
             "time": paste_times,
             "value": np.zeros(lpaste)}
        ).reset_index(drop=True)], axis=1)
    return pd.concat([paste_df, combined])


def mad(data, axis=None):
    return np.mean(np.absolute(data - np.median(data, axis)), axis)


def standardize_trace(trace, min_time=-100, max_time=300, step=0.05):
    cur_times = trace["time"]
    cur_meta = trace.iloc[:1, :5]

    if np.min(cur_times) > min_time:
        prepend_times = np.arange(min_time, np.min(cur_times), step)
        trace = merge_meta(cur_meta, trace, cur_times, prepend_times)

    if np.max(cur_times) < max_time:
        postpend_times = np.arange(np.max(cur_times), max_time, step)
        trace = merge_meta(cur_meta, trace, cur_times, postpend_times)

    v = trace["value"]
    trace["value"] = (v - np.mean(v)) / mad(v)
    return trace.loc[
        (trace["time"] > min_time) &
        (trace["time"] < max_time)
    ]

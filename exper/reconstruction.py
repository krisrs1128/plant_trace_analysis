"""Correlate Traces with Wavelet Basis
"""

import pandas as pd
import numpy as np
import os
import src.preprocess as pre
import src.wavelet as wv
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import SparsePCA
from scipy.interpolate import griddata
from plotnine import *

raw_path = os.path.join("data", "raw", "raw data")
fnames = os.listdir(raw_path)
fnames = [f for f in fnames if f != "msl5 ch1L8 ch2LD ch3L13 ch4L7.txt"] # has no cutpoint

metadata = pre.process_names(fnames)
fnames = [os.path.join(raw_path, s) for s in metadata["fname"]]
metadata["fname"] = fnames
traces = pre.read_traces(fnames)
combined = pd.merge(metadata, traces, "right")
combined["source"] = pd.Categorical(
    combined["source"],
    categories=["L" + str(i) for i in range(20)]
)

## Align traces according to cutpoints
cut_times = combined.loc[combined["cut_point"] == "cut", ]
(ggplot(cut_times) +
 geom_histogram(aes(x = "time", fill = "genotype"), binwidth = 10) +
 facet_grid(["genotype", "."])
 )

combined = combined.loc[combined["cut_point"] == "no_cut", :]
combined = pd.concat([cut_times, combined.iloc[::50, :]])

for file in np.unique(fnames):
    for channel in ["ch1", "ch2", "ch3", "ch4"]:
        print("Aligning " + file + "| channel " + channel)
        cur_trace = (combined["fname"] == file) & (combined["channel"] == channel)
        combined.loc[cur_trace] = pre.align_trace(combined.loc[cur_trace])

(ggplot(combined) +
    geom_point(
        aes(
            x = "time",
            y = "value",
            color = "target"
        ),
        size = 0.01
    ) +
    facet_grid(["source", "genotype"], scales="free_y")
)

## extend / truncate times and standardize
x = []
for file in np.unique(fnames):
    for channel in ["ch1", "ch2", "ch3", "ch4"]:
        print("Standardizing " + file + "| channel " + channel)
        cur_trace = (combined["fname"] == file) & (combined["channel"] == channel)
        x.append(standardize_trace(combined.loc[cur_trace]))

del combined
combined = pd.concat(x)

(ggplot(combined) +
    geom_point(
        aes(
            x = "time",
            y = "value",
            color = "target"
        ),
        size = 0.01
    ) +
    facet_grid(["source", "genotype"], scales="free_y")
)

## build a coefficient matrix
times = np.arange(-90, 290, 0.05)
H = wavelet_basis(times, resolution=2 ** 11)

wv_coefs = []
y_hat = []
for file in np.unique(fnames):
    for channel in ["ch1", "ch2", "ch3", "ch4"]:
        print("Decomposing " + file + "| channel " + channel)
        cur_trace = (combined["fname"] == file) & (combined["channel"] == channel)
        v = griddata(
            combined.loc[cur_trace, "time"].values,
            combined.loc[cur_trace, "value"].values,
            times
        )
        coef, pred = wavelet_coefs(H, v, 0.000001)
        wv_coefs.append({
            "file": file,
            "channel": channel,
            "coef": coef
        })
        y_hat.append({
            "file": file,
            "channel": channel,
            "y_hat": pred,
            "y": v
        })


## get columns which are nonzero in at least one trace
wv_mat = np.zeros((len(wv_coefs), len(wv_coefs[0]["coef"])))
files = []
channels = []
for i, coef in enumerate(wv_coefs):
    wv_mat[i, :] = coef["coef"]
    files.append(coef["file"])
    channels.append(coef["channel"])

wv_mat = wv_mat[:, np.where(wv_mat.any(axis=0))[0]]

wv_df = pd.DataFrame({"fname": files, "channel": channels})
wv_df = pd.merge(wv_df, metadata)
pca_wv = SparsePCA(n_components=4, ridge_alpha=2, alpha=1).fit(wv_mat)
scores = pca_wv.transform(wv_mat)

wv_df = pd.DataFrame({
    "x": scores[:, 0],
    "y": scores[:, 1],
    "z": scores[:, 2],
    "fname": files,
    "channel": channels
})
wv_df = pd.merge(wv_df, metadata)

(ggplot(wv_df) +
 geom_point(
     aes(x="x", y="y", size="z", color="genotype", shape="target")) +
 scale_size_continuous(range=(0.3, 0.7)) +
 # ylim(-0.7, 0.55) +
 # xlim(-1.5, 2.6) +
 facet_wrap("source")
)

## random forest can classify well
rf_model = RandomForestClassifier()
rf_fit = rf_model.fit(wv_mat, wv_df.genotype)
np.argsort(rf_fit.feature_importances_)

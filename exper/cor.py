"""Correlate Traces with Wavelet Basis
"""

import pandas as pd
import numpy as np
import os
import src.preprocess as pre
import src.wavelet as wv
from sklearn.linear_model import Lasso
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
# (ggplot(cut_times) +
#  geom_histogram(aes(x = "time", fill = "genotype"), binwidth = 10) +
#  facet_grid(["genotype", "."])
#  )

combined = combined.loc[combined["cut_point"] == "no_cut", :]
combined = pd.concat([cut_times, combined.iloc[::50, :]])

for file in np.unique(fnames):
    for channel in ["ch1", "ch2", "ch3", "ch4"]:
        print("Aligning " + file + "| channel " + channel)
        cur_trace = (combined["fname"] == file) & (combined["channel"] == channel)
        combined.loc[cur_trace] = pre.align_trace(combined.loc[cur_trace])

# (ggplot(combined) +
#     geom_point(
#         aes(
#             x = "time",
#             y = "value",
#             color = "target",
#         ),
#         size = 0.01
#     ) +
#     facet_grid(["source", "genotype"]) +
#     ylim(-0.07, 0.07)
# )

combined0 = copy.copy(combined)

## extend / truncate times and standardize
x = []
for file in np.unique(fnames):
    for channel in ["ch1", "ch2", "ch3", "ch4"]:
        print("Standardizing " + file + "| channel " + channel)
        cur_trace = (combined["fname"] == file) & (combined["channel"] == channel)
        x.append(pre.standardize_trace(combined.loc[cur_trace]))

del combined
combined = pd.concat(x)

# (ggplot(combined) +
#     geom_point(
#         aes(
#             x = "time",
#             y = "value",
#             color = "target"
#         ),
#         size = 0.01
#     ) +
#     facet_grid(["source", "genotype"], scales="free_y")
# )

## build a coefficient matrix
times = np.unique(combined["time"])
wv_coefs = []
y_hat = []
for file in np.unique(fnames):
    for channel in ["ch1", "ch2", "ch3", "ch4"]:
        print("Decomposing " + file + "| channel " + channel)
        cur_trace = (combined["fname"] == file) & (combined["channel"] == channel)
        H = wv.wavelet_basis(combined.loc[cur_trace, "time"], resolution=2 ** 11)
        coef, pred = wavelet_coefs(H, combined.loc[cur_trace, "value"])
        wv_coefs.append({
            "file": file,
            "channel": channel,
            "coef": coef
        })
        y_hat.append({
            "file": file,
            "channel": channel,
            "y_hat": pred
        })


## get columns which are nonzero in at least one trace

## PCA the coefficient matrix

## plot the scores against the mutation type / source / target

## inspect a couple traces by hand
plt.figure()
H_sub = H[:, np.where(beta != 0.)][:, 0, :]
for i in range(H_sub.shape[1]):
    plt.plot(H_sub[:, i])

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

combined0 = combined

## extend / truncate times and standardize
for file in np.unique(fnames):
    for channel in ["ch1", "ch2", "ch3", "ch4"]:
        print("Standardizing " + file + "| channel " + channel)
        cur_trace = (combined["fname"] == file) & (combined["channel"] == channel)
        combined.loc[cur_trace] = pre.standardize_trace(combined.loc[cur_trace])


# (ggplot(combined) +
#     geom_point(
#         aes(
#             x = "time",
#             y = "value",
#             color = "target",
#         ),
#         size = 0.01
#     ) +
#     facet_grid(["source", "genotype"], scales="free_y")
# )

## build a coefficient matrix

## get columns which are nonzero in at least one trace

## PCA the coefficient matrix

## plot the scores against the mutation type / source / target

## inspect a couple traces by hand
sample_name = "data/raw/raw data/wt6 ch1L13 ch2L8 ch3LD ch4L7.txt"
sample = combined.loc[
    (combined["fname"] == sample_name) &
    (combined["target"] == "L13")
]

y0 = sample.loc[::10, "value"]
y = (y0 - np.mean(y0)) / np.std(y0)
times = sample.loc[::10, "time"]
H = wv.wavelet_basis(times, resolution = 2 ** 12)

lasso_model = Lasso(fit_intercept=False, alpha=0.0001)
fit = lasso_model.fit(H, y)
y_hat = fit.predict(H)
plt.plot(y.values)
plt.plot(y_hat)

beta = fit.coef_
plt.figure()
H_sub = H[:, np.where(beta != 0.)][:, 0, :]
for i in range(H_sub.shape[1]):
    plt.plot(H_sub[:, i])

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

CM = 1 / 2.54
PRX_SINGLE = 8.5 * CM  # ~3.35 in
PRX_ONEHALF = 14.0 * CM  # ~5.51 in
PRX_DOUBLE = 17.8 * CM  # ~7.01 in


def prx_figsize(width="single", aspect=1.5):
    w = {"single": PRX_SINGLE, "1.5": PRX_ONEHALF, "double": PRX_DOUBLE}[width]
    return (w, w * aspect)

plt.style.use("../prx_quantum.mplstyle")

data = pd.read_csv("data.csv")

sizes = data["Size"].values
mh_acorrs = data["MH Auto-corr"].values
gibbs_acorrs = data["Gibbs Auto-corr"].values

plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=0.7)
plt.figure()
plt.plot(4 * sizes, mh_acorrs, label="mh", marker=".")
plt.plot(4 * sizes, gibbs_acorrs, label="gibbs", marker=".")
plt.legend()
plt.xlabel("Total number of spins")
plt.ylabel("Mean autocorrelation time")
plt.xscale("log")
plt.yscale("log")
plt.savefig("ac_times.pdf")
plt.close()
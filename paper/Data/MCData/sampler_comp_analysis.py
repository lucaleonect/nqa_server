import numpy as np
import time
import matplotlib.pyplot as plt

CM = 1 / 2.54
PRX_SINGLE = 8.5 * CM  # ~3.35 in
PRX_ONEHALF = 14.0 * CM  # ~5.51 in
PRX_DOUBLE = 17.8 * CM  # ~7.01 in


def prx_figsize(width="single", aspect=1.5):
    w = {"single": PRX_SINGLE, "1.5": PRX_ONEHALF, "double": PRX_DOUBLE}[width]
    return (w, w * aspect)


plt.style.use("prx_quantum.mplstyle")

def main():
    sizes = np.array([5, 10, 15, 20, 25, 30, 50])
    mh_list = []
    gibbs_list = []

    data = dict(np.load("mcdata.npz", allow_pickle=True))
    for L in sizes:
        mh_temp = np.array(data[str(L)].item()["mh"]["actime"])
        print(mh_temp.max(axis=1).mean())
        gibbs_temp = np.array(data[str(L)].item()["gibbs"]["actime"])
        print(gibbs_temp.max(axis=1).mean())

        mh_list += [mh_temp.max(axis=1).mean()]
        gibbs_list += [gibbs_temp.max(axis=1).mean()]

    plt.rcParams["figure.figsize"] = prx_figsize("single", aspect=1.)
    plt.figure()
    plt.plot(4 * sizes, mh_list, label="mh", marker=".")
    plt.plot(4 * sizes, gibbs_list, label="gibbs", marker=".")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.title("Autocorrelation times")
    plt.xlabel("Total number of spins")
    plt.ylabel("Mean autocorrelation time")
    plt.savefig("ac_times.pdf")
    plt.close()


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

data = np.load("./MCMC/mcdata.npz", allow_pickle=True)

# Convert to csv and save to ../Figures/Figure6/data.csv
save_path = "../Figures/Figure6/data.csv"

sizes = np.array([5, 10, 15, 20, 25, 30, 50])
mh_list = []
gibbs_list = []

for L in sizes:
    mh_temp = np.array(data[str(L)].item()["mh"]["actime"])
    print(mh_temp.max(axis=1).mean())
    gibbs_temp = np.array(data[str(L)].item()["gibbs"]["actime"])
    print(gibbs_temp.max(axis=1).mean())

    mh_list += [mh_temp.max(axis=1).mean()]
    gibbs_list += [gibbs_temp.max(axis=1).mean()]

df = pd.DataFrame({
    "Size": sizes,
    "MH Auto-corr": mh_list,
    "Gibbs Auto-corr": gibbs_list,
})
df.to_csv(save_path, index=False)
import numpy as np

Q = np.load("./matrix_Q_j5_op5_pruned.npy")
n = Q.shape[0]

# Save Q as CSV
np.savetxt("./matrix_Q_j5_op5_pruned.csv", Q, delimiter=",")
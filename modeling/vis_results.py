import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

df = pd.read_csv("./evaluation_data.csv")

y_true = df["linear:y_true"].to_numpy()
y_linear = df["linear:y_pred"].to_numpy()
y_ridge = df["ridge:y_pred"].to_numpy()
y_kernel_ridge=df["kernel_ridge:y_pred"].to_numpy()
t = df["Timestamp"].to_numpy()



plt.plot(y_true[16600:17000])
plt.plot(y_linear[16600:17000])
plt.plot(y_ridge[16600:17000])
plt.plot(y_kernel_ridge[16600:17000])
plt.legend(["true", "linear", "ridge", "kernel_ridge"])
plt.gca().get_xaxis().set_ticks([])

plt.show()

# error = abs(y_kernel_ridge - y_true)


# z_scores = stats.zscore(error)
# z_absoulute = np.absolute(z_scores)
# print(np.mean(z_absoulute))
# print(np.median(z_absoulute))

# plt.hist(z_scores, bins=np.arange(0,5,0.1))
# plt.show()

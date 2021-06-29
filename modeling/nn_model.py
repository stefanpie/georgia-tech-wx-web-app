import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, Conv1D, Input
# from tensorflow.keras.activations import tanh, sigmoid, relu
# from tcn import TCN, tcn_full_summary


df = pd.read_csv("./data/data_processed.csv", nrows=10)
print(df.columns.tolist())

c_x = df.columns.tolist()
c_x = [c for c in c_x if "Timestamp" not in c]
c_x = [c for c in c_x if "+" not in c]

x = df[c_x]

c_y = ["Rain Rate_t+5"]
y = df[c_y]

timestamps = df["Timestamp"]
print(timestamps)

print(x)

# x = x.to_numpy()
# y = y.to_numpy()

# x_train, x_test, y_train, y_test, timestamp_train, timestamp_test = train_test_split(
#     x, y, timestamps, test_size=0.3, random_state=42)\

# x_train = x_train[:5000]
# y_train = y_train[:5000]
# print(x_train.shape)

# x_test = x_test[:1000]
# y_test = y_test[:1000]

# scaler = StandardScaler()
# scaler.fit(x_train)

# x_train_scaled = scaler.transform(x_train)
# x_test_scaled = scaler.transform(x_test)
# x_scaled = scaler.transform(x)



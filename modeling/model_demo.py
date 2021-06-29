import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics.pairwise import polynomial_kernel


from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score,
mean_absolute_percentage_error
)

df = pd.read_csv("./data/data_processed.csv")

c_x = df.columns.tolist()
c_x = [c for c in c_x if "Timestamp" not in c]
c_x = [c for c in c_x if "+" not in c]  

x = df[c_x]

c_y = ["Rain Rate+5"]
y = df[c_y]

timestamps =  df["Timestamp"]
print(timestamps)

x = x.to_numpy()
y = y.to_numpy()

x_train, x_test, y_train, y_test, timestamp_train, timestamp_test = train_test_split(x, y, timestamps, test_size=0.3, random_state=42)


x_train = x_train[:5000]
y_train = y_train[:5000]
print(x_train.shape)

x_test = x_test[:1000]
y_test = y_test[:1000]

scaler = StandardScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_scaled = scaler.transform(x)

models = {
            "linear": LinearRegression(),
            "ridge": Ridge(),
            "kernel_ridge": KernelRidge(kernel="poly"),
            # "svr": SVR(),
            # "gbr": GradientBoostingRegressor()
         }

print("### Fitting Models ###")
for model_name, model in models.items():
    print(f"Fitting: {model_name}")
    model.fit(x_train_scaled, y_train.reshape(-1, 1))
print("Done")
print()


# def three_sigma_error(y_true, y_pred):
#     return float(np.std(y_true-y_pred)*3)

metrics = {"r2": r2_score,
           "mae": mean_absolute_error,
           "mse": mean_squared_error}

evaluation_metrics = {}
evaluation_data = pd.DataFrame()
evaluation_data["Timestamp"] = df["Timestamp"]

print("### Evaluating Models ###")
for model_name, model in models.items():

    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    y_pred = model.predict(x_scaled)
    evaluation_data[f"{model_name}:y_true"] = y
    evaluation_data[f"{model_name}:y_pred"] = y_pred

    evaluation_metrics[model_name] = {}
    for metric_name, metric in metrics.items():
        evaluation_metrics[model_name][metric_name] = {}

        error_train = metric(y_train, y_train_pred)
        error_test = metric(y_test, y_test_pred)

        evaluation_metrics[model_name][metric_name]["train"] = error_train
        evaluation_metrics[model_name][metric_name]["test"] = error_test
        
        print(f"{model_name}:{metric_name}:train:{error_train}")
        print(f"{model_name}:{metric_name}:test:{error_test}")
    print()
evaluation_data.to_csv("evaluation_data.csv", index=False)
print("Done")
print()

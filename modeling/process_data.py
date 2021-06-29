import pandas as pd
from joblib import Parallel, delayed

pd.set_option('display.max_rows', 1000)

df = pd.read_csv("./data/data.csv")


drop_columns = ["Barometer Tendency",
                "Condensation / Dew",
                "Ground Moisture",
                "Ground Temperature",
                "Indoor Pool Air Temperature",
                "Indoor Pool Relative Humidity",
                "Stamps Rec Fields",
                "UV Radiation Sensor"]
df = df.drop(columns=drop_columns)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")


grouped_dfs = df.groupby(df["Timestamp"].diff().ne(pd.Timedelta(hours=1)).cumsum())
time_chunks_dfs = []
for name, group_df in grouped_dfs:
    time_chunks_dfs.append(group_df)

def calculate_time_shifted_features(x):
    hours_foward = 24
    hours_backward = 24*3

    foward_deltas = [-i for i in range(1,hours_foward+1)]
    backward_deltas = [i for i in range(0,hours_backward+1)]
    shifts = backward_deltas + foward_deltas

    d_t = 1
    features = list(x.columns)
    for s in shifts:
        shift_label = "_t"
        if s > 0:
            shift_label += str(-1*d_t*s)
        else:
            shift_label += "+" + str(-1*d_t*s)

        f_p = [f + shift_label for f in features]
        x[f_p] = x[features].shift(s)

    x = x.dropna(axis=0)
    return x

print("#### Cleaning and Feature Extraction ####")
print(f"Processing {len(time_chunks_dfs)} time chunks...")

print("Calculating time shifted feature...")
# time_chunks_dfs = list(map(calculate_time_shifted_features, time_chunks_dfs))
time_chunks_dfs = Parallel(n_jobs=-1,verbose=0)(delayed(calculate_time_shifted_features)(x) for x in time_chunks_dfs)

print("Done")
print()


print("#### Combining Data Back Togeather ####")
time_chunks_dfs = [x for x in time_chunks_dfs if not x.empty] 
final_df = pd.concat(time_chunks_dfs)
final_df = final_df.sort_values(by=["Timestamp"])
print("Done")
print()


print("#### Saving processed data to file ####")
final_df.to_csv('./data/data_processed.csv', index = False)
print("Done")
print()

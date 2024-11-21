from os import system
from pathlib import Path as P
from time import sleep
from pprint import pprint

import pandas as pd


results = list(P().glob("training/*/results.csv"))
print(f"{len(results)} training folders:")
pprint(results)

# For each csv, load only the last row
dfs = []
for r in results:
    df = pd.read_csv(r, index_col=0)
    # Remove every column that is not a metric
    df = df.filter(regex="metrics")
    # Before adding the cv, add a column for the F1 score
    df["F1"] = (
        2
        * df["   metrics/precision(B)"]
        * df["      metrics/recall(B)"]
        / (df["   metrics/precision(B)"] + df["      metrics/recall(B)"])
    )
    # Add a column for the name of the run
    df["run"] = r.parent.name
    dfs.append(df.iloc[-1])
df = pd.concat(dfs, axis=1).T

# Sort and show results
df_f1 = df.sort_values("F1", ascending=False).head(n=5)
print()
print("Runs sorted by f1:")
print(df_f1)

df_map50 = df.sort_values("       metrics/mAP50(B)", ascending=False).head(n=5)
print()
print("Results sorted by map50:")
print(df_map50)

print()
print("Results sorted by precision:")
print(df.sort_values("   metrics/precision(B)", ascending=False).head(n=5))

print()
print("Results sorted by recall:")
print(df.sort_values("      metrics/recall(B)", ascending=False).head(n=5))


# Write the best 5 runs from df_f1['run'] to a text file
with open("best_f1.txt", "w") as f:
    for run in df_f1["run"]:
        f.write(run + "\n")

# with open("./best_f1.txt", "r") as f:
#     best_f1 = f.read().split("\n")[:-1]
# best_f1 = [run for run in df[]]

test_folders = [P("./training") / bf for bf in df_f1["run"]]
assert all([tf.exists() for tf in test_folders]), "Some test folders do not exist."

for i, tf in enumerate(test_folders):
    run = tf.name
    c = f"python test.py -m .\\{tf}\\weights\\best.pt -y .\\data.yaml -p testing -s test -b 256 -si -n {run} -d 0,1,2,3"
    print(f"================ Testing run {i+1} named {run} ==================")
    system(c)
    print()
    sleep(1)

import pandas as pd
from io import StringIO

data = """
Class Images Instances P R mAP50 mAP50-95
all 1737 20028 0.761 0.398 0.586 0.378
Air_basket 1737 355 1 0.586 0.793 0.562
Belt 1737 361 0.228 0.0859 0.124 0.039
Boot 1737 616 0.775 0.151 0.459 0.169
Face 1737 2082 0.931 0.842 0.908 0.707
Glove 1737 819 0.68 0.228 0.467 0.265
Goggles 1737 357 0.8 0.0224 0.411 0.205
Hand 1737 2623 0.895 0.758 0.846 0.516
Head 1737 3956 0.947 0.883 0.933 0.781
Helmet 1737 1795 0.909 0.503 0.721 0.447
Hood 1737 778 0.921 0.15 0.535 0.314
Ladder 1737 100 0.75 0.03 0.386 0.14
Pants 1737 2436 0.846 0.823 0.863 0.611
Shield 1737 102 0 0 0 0
Shirt 1737 3545 0.939 0.86 0.92 0.711
Sleeves 1737 103 0.8 0.0388 0.419 0.21
"""

df = pd.read_csv(StringIO(data), sep=" ")
df["F1"] = 2 * (df["P"] * df["R"]) / (df["P"] + df["R"])
df["F1"].fillna(0, inplace=True)

with open("class_results.md", "w") as f:
    f.write(df.to_markdown())

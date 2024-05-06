from os import system
from pathlib import Path as P
from time import sleep

test_folders = list(P("./training/").iterdir())
for i, tf in enumerate(test_folders):
    run = tf.name
    c = f"python test.py -m .\\{tf}\\weights\\best.pt -y .\\data.yaml -p testing -s test -b 256 -si -n {tf.name} -d 0,1,2,3"
    print(f"================ Testing run {i+1} named {run} ==================")
    system(c)
    print()
    sleep(1)
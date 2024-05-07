from os import system
from pathlib import Path as P
from time import sleep
from pprint import pprint

with open("./best_f1.txt", "r") as f:
    best_f1 = f.read().split("\n")[:-1]
    
test_folders = [P("./training") / bf for bf in best_f1]
assert all([tf.exists() for tf in test_folders]), "Some test folders do not exist."

for i, tf in enumerate(test_folders):
    run = tf.name
    c = f"python test.py -m .\\{tf}\\weights\\best.pt -y .\\data.yaml -p testing -s test -b 256 -si -n {run} -d 0,1,2,3"
    print(f"================ Testing run {i+1} named {run} ==================")
    system(c)
    print()
    sleep(1)
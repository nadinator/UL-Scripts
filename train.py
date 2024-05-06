#! C:\Users\Nadim\.conda\envs\ultra10\python.exe

from time import sleep, time
from itertools import product

from ultralytics import YOLO


data_yaml = "./data.yaml"
model = YOLO("yolov8l.pt") #* Should be modified per experiment.
training_args = {  # * Should be modified per experiment.
    "data": str(data_yaml),
    "dropout": 0.2,
    "patience": 250,
    "batch": 256,
    "cache": True,
    "device": "0,1,2,3",
    "val": True,
    "workers": 0,  #! Only increase this on Windows if you want RAM to explode.
    "epochs": 750,
    "project": "./training/",
    "plots": True,
    "pretrained": True,
    "optimizer": "AdamW",
    # "resume": True
}
tuning_space = {  # * Should be modified per experiment.
    "lr0": (1e-9, 1e-7, 1e-5),
    "lrf": (0.0001, 0.001),
    "momentum": (0.95, 0.9),
    # "pretrained": (False, True),
    # "optimizer": ("AdamW", "SGD"),
}


if __name__ == "__main__":
    skip = 0
    grid = list(product(*tuning_space.values(), repeat=1))[skip:]
    keys = list(tuning_space.keys())

    for j, pair in enumerate(grid):
        name_parts = [f"{keys[i]}{pair[i]}" for i in range(len(pair))]
        training_args["name"] = f"run{j}--" + "_".join(name_parts)

        tuning_args = dict(zip(keys, pair))
        merged_args = {**training_args, **tuning_args}

        t0 = time()
        result = model.train(**merged_args)
        tf = time()

        print(f"========== RESULT FOR {training_args['name']}==========")
        print(f"Time taken: {tf-t0:.2f} seconds")
        if result is not None:
            precision = result.results_dict["metrics/precision(B)"]
            recall = result.results_dict["metrics/recall(B)"]
            f1 = 2*precision*recall / (precision+recall)
            print(f"Overall F1 score: {f1:.2f}")
            # print(result)
        print("========== END RESULT ==========")
        sleep(1)

#! C:\Users\Nadim\.conda\envs\ultra10\python.exe

from time import sleep, time
from itertools import product

from ultralytics import YOLO


data_yaml = "./data.yaml"
model = YOLO("yolov8l.pt")
training_args = {
    "data": str(data_yaml),
    "dropout": 0.2,
    "patience": 250,
    "batch": 128,
    "cache": True,
    "device": "2,3",
    "val": True,
    "workers": 32,
    "epochs": 750,
    "project": "./e5-training/",
    "plots": True,
    # "pretrained": True,
    # "resume": True
}
tuning_space = {
    "lr0": (1e-9, 1e-7, 1e-5),
    "lrf": (0.0001, 0.001),
    "momentum": (0.95, 0.9),
    "pretrained": (False, True),
    "optimizer": ("AdamW", "SGD"),
}

if __name__ == "__main__":
    grid = product(tuning_space.values(), repeat=1)
    for  values in grid:
        name_parts = [f"{key}{value}" for key, value in tuning_args.items()]
        training_args["name"] = "_".join(name_parts)
        
        tuning_args = dict(zip(tuning_space.keys(), values))
        merged_args = {**training_args, **tuning_args}

        t0 = time()
        result = model.train(**merged_args)
        tf = time()

        #? For some reason, `result` is None here but not when I run YOLO().train() straight
        # precision = result.results_dict["metrics/precision(B)"]
        # recall = result.results_dict["metrics/recall(B)"]
        # f1 = 2*precision*recall / (precision+recall)

        print(f"========== RESULT FOR {training_args['name']}==========")
        print(f"Time taken: {tf-t0:.2f} seconds")
        # print(f"Overall F1 score: {f1:.2f}")
        print(result)
        print("========== END RESULT ==========")

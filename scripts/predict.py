# Written by Mohammed Elwaleed, edited by Nadim Bou Alwan.

import argparse
import os 
import tqdm
import sys

from torch import cuda
from ultralytics import YOLO
from pathlib import Path as P


def main():
    parser = argparse.ArgumentParser(description='Run inferance of a YOLOv8 model on your test split (or any as a matter of fact).')
    parser.add_argument("--model_path", type=str, help="Path to best.pt")  #!
    parser.add_argument("--save_path", type=str, help="Path to predictions folder")  #!
    parser.add_argument('--device', type=str, default="0", help='Devices')
    parser.add_argument("--conf", type=float, default=0.3, help="conf")
    parser.add_argument("--images_path", type=str, help="name", default=None)  #!
    args = parser.parse_args()

    device = "cpu" if not cuda.is_available() else args.device
    model = YOLO(args.model_path)

    save_path = P(args.save_path) / "predictions"
    save_path.mkdir(exist_ok=True)
    images_path = P(args.images_path)

    for im in tqdm.tqdm(images_path.iterdir(), desc="Predicting on images..."):
        result = model(str(im.resolve()), verbose=False, conf=args.conf, device=device)
        annotated_frame = result[0]
        annotated_frame.save(filename=str(save_path / im.name))

    print("Done!")


if __name__ == '__main__':
    main()

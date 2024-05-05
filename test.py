import os
import sys
import yaml
import argparse
from pathlib import Path

import cv2
import tqdm
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


NAMES = [
    "Air_basket",
    "Belt",
    "Boot",
    "Face",
    "Foot",
    "Glove",
    "Goggles",
    "Hand",
    "Head",
    "Helmet",
    "Hood",
    "Ladder",
    "Pants",
    "Shield",
    "Shirt",
    "Sleeves",
]


def get_labels_from_file(path, remove_class=False):
    """Reads labels from a file and returns them as a list."""
    labels = []
    if path.exists():
        with path.open("r") as f:
            labels = [line.strip().split(" ") for line in f.readlines()]
        # clean the format
        for i in range(len(labels)):
            labels[i] = [float(j) for j in labels[i]]
            labels[i][0] = int(labels[i][0])  # class needs to be int
            if remove_class:
                labels[i].pop(0)
    return labels


def cxcywh2xyxy(bbox):
    """Converts bounding box format from center to corners."""
    box_center_x, box_center_y, box_width, box_hight = bbox
    box_min_x = box_center_x - box_width / 2
    box_max_x = box_center_x + box_width / 2
    box_min_y = box_center_y - box_hight / 2
    box_max_y = box_center_y + box_hight / 2
    return [box_min_x, box_min_y, box_max_x, box_max_y]


def main():
    """Main function to run inference of a YOLOv8 model on your test split."""
    parser = argparse.ArgumentParser(description="Run inference of a YOLOv8 model on your test split (or any as a matter of fact).")

    parser.add_argument("-m", "--model_path", type=str, help="Path to best.pt")
    parser.add_argument("-y", "--yaml_path", type=str, help="Path to data.yaml")
    parser.add_argument("-p", "--project", type=str, help="project name")
    parser.add_argument("-n", "--name", type=str, help="name")
    parser.add_argument("-s", "--split", type=str, help="Split to evaluate over [train,val,test]")
    parser.add_argument("-d", "--device", type=str, default="0", help="Devices")
    parser.add_argument("-b", "--batch", type=int, default=64, help="batch size")
    parser.add_argument("-c", "--conf", type=float, default=0.5, help="conf")
    parser.add_argument("-si", "--save_images", dest="save_images", action="store_true", help="Save bounding boxes")
    parser.add_argument("-ag", "--add_gt", dest="add_gt", action="store_true", help="Save gt bounding boxes")
    parser.add_argument("-ip", "--images_path", type=str, help="name", default=None)

    parser.set_defaults(add_gt=False)
    parser.set_defaults(save_images=False)

    # Load parameters
    args = parser.parse_args()
    model_path = Path(args.model_path)
    yaml_path = Path(args.yaml_path)

    # Load yaml file data
    with yaml_path.open() as f:
        yaml_data = yaml.safe_load(f)

    # Do validation
    model = YOLO(model_path)
    validation_results = model.val(
        data=yaml_path,
        imgsz=640,
        batch=args.batch,
        conf=args.conf,
        iou=0.7,
        device=args.device,
        split=args.split,
        project=args.project,
        name=args.name,
        workers=0,
    )

    p = validation_results.results_dict["metrics/precision(B)"]
    r = validation_results.results_dict["metrics/recall(B)"]
    f1 = 2 * p * r / (p + r)
    print()
    print("Average F1 Score: ", f1)
    print("Average Precision: ", p)
    print("Average Recall: ", r)
    print("Average mAP: ", validation_results.results_dict["metrics/mAP50(B)"])
    print()

    # Draw on and save images
    if args.save_images:
        save_path = Path(validation_results.save_dir) / "predictions"
        save_path.mkdir()

        # Try to read images+labels from yaml file if args.images_path was not passed in
        if args.images_path is None:
            data_path = Path(yaml_data["data"])
            images_path = data_path / "images" / args.split
            labels_path = data_path / "labels" / args.split
        else:
            images_path = Path(args.images_path)
            labels_path = images_path.with_name("labels")

        assert images_path.exists(), f"Image path does not exist: {images_path}"
        assert labels_path.exists(), f"Labels path does not exist: {labels_path}"

        # Iterate over images, draw boxes on them, and save them
        for image in tqdm.tqdm(list(images_path.iterdir()), desc="Saving prediction images..."):
            result = model(str(image.resolve()), verbose=False, conf=args.conf)
            annotated_frame = result[0]

            if args.add_gt:
                original_image = cv2.imread(str(image.resolve()))
                h, w, c = original_image.shape

                annotator = Annotator(original_image)

                gt_labels = get_labels_from_file(labels_path / (image.name + "txt"))
                for box in gt_labels:
                    box[1] *= w
                    box[3] *= w
                    box[2] *= h
                    box[4] *= h
                    b = cxcywh2xyxy(box[1:])
                    c = box[0]
                    annotator.box_label(b, NAMES[int(c)])

                gt_annotated_fram = annotator.result()
                final_frame = np.hstack(
                    (gt_annotated_fram, annotated_frame.plot())
                )
                final_frame = Image.fromarray(final_frame)
                final_frame.save(save_path / image.name)
            else:
                annotated_frame.save(filename=save_path / image.name)

    # save the command into (command.txt)
    s = " ".join(sys.argv)
    with open(f"{str(validation_results.save_dir)}/command.txt", "w") as f:
        f.write(os.path.basename(sys.executable) + " " + s)

    print("Done!")


if __name__ == "__main__":
    main()

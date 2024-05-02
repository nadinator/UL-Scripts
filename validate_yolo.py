from ultralytics import YOLO
import argparse
import os 
import tqdm
import sys
import cv2
from ultralytics.utils.plotting import Annotator 
import numpy as np
from PIL import Image

def get_labels_from_file(path, remove_class=False):
    if os.path.exists(path) :
        with open(path, 'r') as f:
            labels = f.readlines()
        # clean the formate
        for i,_ in enumerate(labels):
            labels[i] = labels[i].replace('\n',"").split(' ')
            labels[i] = [float(j) for j in labels[i]]
            labels[i][0] = int(labels[i][0]) # class need to be int 
            if remove_class:
                labels[i].pop(0)

    else:
        labels = []
    
    return labels

def cxcywh2xyxy(bbox):
    box_center_x, box_center_y, box_width , box_hight = bbox
    box_min_x = box_center_x - box_width/2
    box_max_x = box_center_x + box_width/2
    box_min_y = box_center_y - box_hight/2
    box_max_y = box_center_y + box_hight/2
    return [box_min_x,box_min_y, box_max_x, box_max_y]


def main():
    parser = argparse.ArgumentParser(description='Run inferance of a YOLOv8 model on your test split (or any as a matter of fact).')
    parser.add_argument('--model_path', type=str, help='Path to best.pt') #!
    parser.add_argument('--yaml_path' , type=str, help='Path to data.yaml') #!
    parser.add_argument('--project' , type=str, help='project name') #!
    parser.add_argument("--name", type=str, help="name")  #!
    parser.add_argument('--split', type=str, help='Split to evaluate over [train,val,test]') #!
    parser.add_argument("--device", type=str, default="0", help="Devices")  #!
    parser.add_argument("--batch", type=int, default=32, help="batch size")  #!
    parser.add_argument("--conf", type=float, default=0.5, help="conf")  #!
    parser.add_argument('--save_images', dest='save_images', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_images=False)  #!
    parser.add_argument('--add_gt', dest='add_gt', action='store_true', help='Save gt bounding boxes')
    parser.set_defaults(add_gt=False)
    parser.add_argument('--images_path' , type=str, help='name', default=None)

    args = parser.parse_args()

    model = YOLO(args.model_path)
    validation_results = model.val(data=args.yaml_path,
                                   imgsz=640,
                                   batch=args.batch,
                                   conf=args.conf,
                                   iou=0.7,
                                   device=args.device,
                                   split=args.split,
                                   project=args.project,
                                   name=args.name,
                                   workers=0)

    # save images
    if args.save_images:
        print("Saving Prediction images ...")
        save_path = str(validation_results.save_dir) + "/predictions/"
        os.mkdir(save_path)

        if args.images_path is None:
            images_path = args.yaml_path.replace('data.yaml', f'data/{args.split}/images')
            labels_path = args.yaml_path.replace('data.yaml', f'data/{args.split}/labels')
        else:
            images_path = args.images_path
            labels_path = images_path.replace('/images','/labels')

        images = os.listdir(images_path)                  

        names = ['Air_basket', 'Belt', 'Boot', 'Face', 'Foot', 'Glove', 'Goggles',
                 'Hand', 'Head', 'Helmet', 'Hood', 'Ladder', 'Pants', 'Shield', 
                 'Shirt', 'Sleeves'] 

        for image in tqdm.tqdm(images):
            result = model(images_path+"/"+image, verbose=False, conf=args.conf)
            annotated_frame = result[0]

            if args.add_gt:
                original_image = cv2.imread(images_path+"/"+image)
                h,w,c =original_image.shape

                annotator = Annotator(original_image)

                gt_labels= get_labels_from_file(labels_path+"/"+image.replace('jpg','txt'))
                for box in gt_labels:
                    box[1] *=w ; box[3] *=w
                    box[2] *=h ; box[4] *=h
                    b = cxcywh2xyxy(box[1:])
                    c = box[0]
                    annotator.box_label(b, names[int(c)])

                gt_annotated_fram = annotator.result()
                final_frame = np.hstack((gt_annotated_fram,annotated_frame.plot()))
                final_frame = Image.fromarray(final_frame)
                final_frame.save(save_path+image)
            else:        
                annotated_frame.save(filename=save_path+image )

    # save the command into (command.txt)
    s = ""
    for arg in sys.argv:
        s = s + arg + " "

    with open(f'{str(validation_results.save_dir)}/command.txt', 'w') as f:
        f.write(os.path.basename(sys.executable) + " " + s)

    print("Done!")

if __name__ == '__main__':
    main()

# Written by Mohammed Elwaleed, edited by Nadim Bou Alwan.

import argparse
import os 
import tqdm
import sys

from ultralytics import YOLO


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
    parser.add_argument("--model_path", type=str, help="Path to best.pt")  #!
    parser.add_argument("--save_path", type=str, help="Path to best.pt")  #!
    parser.add_argument('--device', type=str, default="0", help='Devices')
    parser.add_argument('--conf', type=float, default=0.5, help='conf')
    parser.add_argument('--save_images', dest='save_images', action='store_true', help='Save bounding boxes') #!
    parser.set_defaults(save_images=False)
    parser.add_argument('--images_path' , type=str, help='name', default=None)  #!

    args = parser.parse_args()

    model = YOLO(args.model_path)

    # save images
    if args.save_images:
        print("Saving Prediction images ...")
        save_path = str(args.save_path) + "/predictions/"
        os.makedirs(save_path)

        images_path = args.images_path
        labels_path = images_path.replace('/images','/labels')

        images = os.listdir(images_path)                  

        names = ['Shirt'] 

        for image in tqdm.tqdm(images):
            result = model(images_path+"/"+image, verbose=False, conf=args.conf)
            annotated_frame = result[0]

            annotated_frame.save(filename=save_path+image )
            break

    # save the command into (command.txt)
    s = ""
    for arg in sys.argv:
        s = s + arg + " "

    with open(f'{str(args.save_path)}/command.txt', 'w') as f:
        f.write(os.path.basename(sys.executable) + " " + s)

    print("Done !")

if __name__ == '__main__':
    main()

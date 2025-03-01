"""
    This file shows how to load and use the dataset
"""

from __future__ import print_function
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


parser = argparse.ArgumentParser(description="Mapillary Vistas demo")
parser.add_argument("--v1_2", action="store_true",
                    help="Demo version 1.2 of the dataset instead of 2.0")


def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array

def main(args):
    version = "v1.2" if args.v1_2 else "v2.0"

    # a nice example
    image_id = 'M2kh294N9c72sICO990Uew'

    # read in config file
    with open('config_{}.json'.format(version)) as config_file:
        config = json.load(config_file)

    labels = config['labels']

    # define road labels
    road_related_labels = [
        "construction--flat--road",               
        "construction--flat--crosswalk-plain",  
        "marking--discrete--crosswalk-zebra",  
        "marking--discrete--arrow--left",    
        "marking--discrete--arrow--right",   
        "marking--discrete--arrow--straight",  
        "marking--discrete--stop-line",  
        "marking--continuous--dashed",  
        "marking--continuous--solid",   
        "marking--discrete--symbol--bicycle", 
        "construction--flat--bike-lane",
        "construction--flat--service-lane"]
    
    # search for ID
    road_label_ids = []
    for label_id, label in enumerate(labels):
        if label["name"] in road_related_labels:
            road_label_ids.append(label_id)
            print(f"Found road-related label: {label['name']} (ID: {label_id})")
            

    if not road_label_ids:
        print("No road-related labels found!")
        return


    # set up paths for every image
    image_path = "training/images/{}.jpg".format(image_id)
    label_path = "training/{}/labels/{}.png".format(version, image_id)

    # load images
    base_image = Image.open(image_path)
    label_image = Image.open(label_path)

    # convert labeled data to numpy arrays for better handling
    label_array = np.array(label_image)

    # for visualization, we apply the colors stored in the config
    colored_label_array = apply_color_map(label_array, labels)


    # create road mask
    road_mask = np.zeros_like(label_array, dtype=bool)

    for label_id in road_label_ids:
        road_mask |= (label_array == label_id)

    road_image = np.array(base_image) * road_mask[:, :, np.newaxis]


    # plot the result
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30,10))

    ax[0].imshow(base_image)
    ax[0].set_title("Base image")
    ax[1].imshow(colored_label_array)
    ax[1].set_title("Training labels")
    ax[2].imshow(road_image)
    ax[2].set_title("Road Region")

    fig.tight_layout()
    fig.savefig('Road_test_plot.png')
        

if __name__ == "__main__":
    main(parser.parse_args())

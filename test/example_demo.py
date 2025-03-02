"""
    This file shows how to load and use the dataset
"""

from __future__ import print_function
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import os


parser = argparse.ArgumentParser(description="Mapillary Vistas demo")
parser.add_argument("--v1_2", action="store_true",
                    help="Demo version 1.2 of the dataset instead of 2.0")
parser.add_argument("--max_images", type=int, default=5, 
                    help="Maximum number of images to process(default: 5)")


def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]
    return color_array


def main(args):
    version = "v1.2" if args.v1_2 else "v2.0"
    max_images = args.max_images

    # a nice example
    #image_id = 'M2kh294N9c72sICO990Uew'

    # read in config file
    with open('config_{}.json'.format(version)) as config_file:
        config = json.load(config_file)
    labels = config['labels']

    # define road related labels
    road_related_labels = [
        "construction--flat--road",
        "construction--flat--sidewalk",               
        "construction--flat--crosswalk-plain",  
        "construction--flat--bike-lane",
        "construction--flat--service-lane",
        "marking--discrete--crosswalk-zebra",  
        "marking--discrete--arrow--left",    
        "marking--discrete--arrow--right",   
        "marking--discrete--arrow--straight",  
        "marking--discrete--symbol--bicycle",
        "marking--discrete--stop-line",  
        "marking--continuous--dashed",  
        "marking--continuous--solid"
        ]
    
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
    images_dir = "training/images"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg")) ]

    if max_images > 0:
        image_files = image_files[:max_images]

    # plot one figure for overview
    fig, axs = plt.subplots(nrows=max_images, ncols=3, figsize=(15, 5 * max_images))
    if max_images == 1:
        axs = axs.reshape(1, -1)

    for idx, image_file in enumerate(image_files):
        image_id = os.path.splitext(image_file)[0]  # remove the extension, get the id
        image_path = os.path.join(images_dir, image_file)
        label_path = "training/{}/labels/{}.png".format(version, image_id)

        # load images
        base_image = Image.open(image_path)
        label_image = Image.open(label_path)

        # convert labeled data to numpy arrays for better handling
        label_array = np.array(label_image)

        # create road mask
        road_mask = np.zeros_like(label_array, dtype=bool)
        for label_id in road_label_ids:
            road_mask |= (label_array == label_id)
        
        # extract the road region
        road_image = np.array(base_image) * road_mask[:, :, np.newaxis]

        ax1, ax2, ax3 = axs[idx] # three columns for each row
        ax1.imshow(base_image)
        ax1.set_title(f"Base image {idx + 1}")
        ax1.axis("off")

        ax2.imshow(apply_color_map(label_array, labels))
        ax2.set_title(f"Training labels {idx + 1}")
        ax2.axis("off")

        ax3.imshow(road_image)
        ax3.set_title(f"Road Region {idx + 1}")
        ax3.axis("off")

    fig.tight_layout()

    output_path = os.path.join(output_dir, "image_overview.png")
    fig.savefig(output_path)
    plt.close(fig) 
    print(f"Saved result for to {output_path}")
            

if __name__ == "__main__":
    main(parser.parse_args())

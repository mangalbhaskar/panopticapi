#!/usr/bin/env python2
'''
Visualization demo for panoptic COCO sample_data

The code shows an example of color generation for panoptic data (with
"generate_new_colors" set to True). For each segment distinct color is used in
a way that it close to the color of corresponding semantic class.

Usage example(s):
python visualization.py --colors \
 --input_json_file $AI_DATA/data/ms-coco/annotations/panoptic_val2017.json \
 --segmentations_folder $AI_DATA/data/ms-coco/panoptic_val2017 \
 --img_folder $AI_DATA/data/ms-coco/val2017


python visualization.py --colors \
 --input_json_file $AI_DATA/data/ms-coco/annotations/panoptic_train2017.json \
 --segmentations_folder $AI_DATA/data/ms-coco/panoptic_train2017 \
 --img_folder $AI_DATA/data/ms-coco/train2017
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from utils import IdGenerator, rgb2id

import argparse


# whether from the PNG are used or new colors are generated
generate_new_colors = False

# input_json_file = './sample_data/panoptic_examples.json'
# segmentations_folder = './sample_data/panoptic_examples/'
# img_folder = './sample_data/input_images/'
# categories_json_file = './panoptic_coco_categories.json'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script converts panoptic COCO format to detection \
         COCO format. See this file's head for more information."
    )
    parser.add_argument('--input_json_file', type=str,
                        help="JSON file with panoptic COCO format",
                        default='./sample_data/panoptic_examples.json')
    parser.add_argument(
        '--segmentations_folder', type=str,
        help="Folder with  panoptic COCO format segmentations. \
        Default: X if input_json_file is X.json",
        default='./sample_data/panoptic_examples/'
    )
    parser.add_argument('--img_folder', type=str,
                        help="Input Image folder",
                        default='./sample_data/input_images/')
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories information",
                        default='./panoptic_coco_categories.json')
    parser.add_argument('--colors', action='store_true',
                        help="whether from the PNG are used or new colors are generated")
    args = parser.parse_args()

    if args.colors:
        generate_new_colors = True
    input_json_file = args.input_json_file
    segmentations_folder = args.segmentations_folder
    img_folder = args.img_folder
    categories_json_file = args.categories_json_file


with open(input_json_file, 'r') as f:
    coco_d = json.load(f)

ann = np.random.choice(coco_d['annotations'])

with open(categories_json_file, 'r') as f:
    categories_list = json.load(f)
categegories = {category['id']: category for category in categories_list}

# find input img that correspond to the annotation
img = None
for image_info in coco_d['images']:
    if image_info['id'] == ann['image_id']:
        try:
            img = np.array(
                Image.open(os.path.join(img_folder, image_info['file_name']))
            )
        except:
            print("Undable to find correspoding input image.")
        break

segmentation = np.array(
    Image.open(os.path.join(segmentations_folder, ann['file_name'])),
    dtype=np.uint8
)
segmentation_id = rgb2id(segmentation)
# find segments boundaries
boundaries = find_boundaries(segmentation_id, mode='thick')

if generate_new_colors:
    segmentation[:, :, :] = 0
    color_generator = IdGenerator(categegories)
    for segment_info in ann['segments_info']:
        color = color_generator.get_color(segment_info['category_id'])
        mask = segmentation_id == segment_info['id']
        segmentation[mask] = color

# depict boundaries
segmentation[boundaries] = [0, 0, 0]

if img is None:
    plt.figure()
    plt.imshow(segmentation)
    plt.axis('off')
else:
    plt.figure(figsize=(9, 5))
    plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(segmentation)
    plt.axis('off')
    plt.tight_layout()
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 17:24:19 2024

@author: becky
"""
"""
This utility module provides a collection of functions and tools for supporting
deep learning and image processing tasks. It includes operations from libraries 
such as PyTorch, OpenCV, Numpy, and Matplotlib, indicating its use in neural 
network operations, image manipulation, and data handling. The module contains 
functions for model loading, data preprocessing, image transformations, and visualization aids. 
These utilities are designed to be reusable and modular, facilitating their integration 
into larger projects or scripts focusing on image analysis.
"""
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from segment_anything import sam_model_registry
from statistics import mean

from tqdm import tqdm
from torch.nn.functional import threshold, normalize
from segment_anything import SamPredictor, sam_model_registry
from collections import defaultdict

from segment_anything.utils.transforms import ResizeLongestSide
import os

# Helper functions provided in https://github.com/facebookresearch/segment-anything/blob/9e8f1309c94f1128a6e5c047a10fdcb02fc8d651/notebooks/predictor_example.ipynb
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def get_bbox_coords(i, list_masks):
    bbox_coords = {}
    for f in list_masks[100*i:100*(i+1)]:
        k = f.stem
        
        im = cv2.imread(f.as_posix())
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        if len(contours) > 1:
          x,y,w,h = cv2.boundingRect(contours[0])
          height, width, _ = im.shape
          bbox_coords[k] = np.array([x, y, x + w, y + h])

    return bbox_coords

def get_ground_truth(bbox_coords, mask_paths):
    ground_truth_masks = {}
    for k in bbox_coords.keys():
        gt_grayscale = cv2.imread(os.path.join(mask_paths, '{}.jpg'.format(k)), cv2.IMREAD_GRAYSCALE)
        ground_truth_masks[k] = (gt_grayscale == 0)
    
    return ground_truth_masks

def image_transform(image, transform, device):
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    return transformed_image

def bbox_transform(transform, prompt_box, original_image_size, device):
    box = transform.apply_boxes(prompt_box, original_image_size)
    box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
    box_torch = box_torch[None, :]
    return box_torch
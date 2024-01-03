# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:33:57 2023

@author: bsoltanian
"""

"""
This Python script is designed for performing automatic image segmentation, 
likely leveraging deep learning models. It imports libraries such as OpenCV, 
PyTorch, and Numpy for image manipulation and neural network operations. 
The core functionality is encapsulated in the `inference` function, which 
processes images from a given directory, applies a segmentation model, 
and saves the results. This script is potentially used for tasks requiring 
precise segmentation in image datasets, such as feature extraction or data 
analysis in various applications.
"""
import os
import cv2
import torch
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import sam_model_registry

    
def inference(mask_generator, image_dir, save_dir):
    res = []
    for file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file)
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = mask_generator.generate(image_rgb)
        res.append(result)
        mask_annotator = sv.MaskAnnotator(color_lookup = sv.ColorLookup.INDEX)
        detections = sv.Detections.from_sam(result)
        annotated_image = mask_annotator.annotate(image_bgr, detections)
        
        color_annotator = sv.ColorAnnotator(color_lookup = sv.ColorLookup.INDEX)
        annotated_frame = color_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        
        pred_name = file.split('.')[0] + '_pred.jpg'
        save_path = os.path.join(save_dir, pred_name)
        cv2.imwrite(save_path, annotated_image)
    return res
        
def show_output(result_dict,axes=None):
     if axes:
        ax = axes
     else:
        ax = plt.gca()
        ax.set_autoscale_on(False)
     sorted_result = sorted(result_dict, key=(lambda x: x['area']),      reverse=True)
     # Plot for each segment area
     for val in sorted_result:
        mask = val['segmentation']
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, mask*0.5)))        

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = 'vit_b'

#model_type = 'vit_b'
CHECKPOINT_PATH = 'model/sam_vit_b_01ec64_9.pth'

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)


mask_generator = SamAutomaticMaskGenerator(sam)

#IMAGE_PATH = 'input_kaggle_original/images/PV03_349422_1168341.jpg'

image_dir = 'data_jpg/test_bmp_only_jpg'
mask_dir = 'data_jpg/masks_jpg'
save_dir = 'data_jpg/result'

res = inference(mask_generator, image_dir, save_dir)

#show_output(res[1], axes=None)
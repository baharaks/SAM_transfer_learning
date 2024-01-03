#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:58:45 2024

@author: becky
"""

"""
This Python script is designed for processing and analyzing solar panel images. 
It utilizes libraries like OpenCV, Numpy, and Matplotlib for image handling and analysis. 
Key functionalities include calculating the center of detected regions and displaying 
image masks for segmentation or analysis. The script likely integrates with the Roboflow 
platform for enhanced image processing capabilities, suggesting its use in advanced solar 
panel image analysis tasks such as defect detection or performance assessment.

"""
import os 
import cv2
import numpy as np
from roboflow import Roboflow
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

def calculate_center(x, y, w, h):
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    # plt.savefig(save_path)
    
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
    
sam_checkpoint = 'model/sam_vit_b_01ec64_9.pth' #"sam_vit_h_4b8939.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

rf = Roboflow(api_key="RPl8XdMUDmesij7OaYNe")

project = rf.workspace().project("aerial-solar-panels")
model = project.version(6).model

image_dir = 'data_jpg/test_bmp_only_jpg'
# mask_dir = 'data_jpg/masks_jpg'
save_dir = 'data_jpg/result_roboflow'
final_dir = 'data_jpg/result_final'

for i, file in enumerate(os.listdir(image_dir)):
    # infer on a local image
    image_path = os.path.join(image_dir, file)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_path = os.path.join(save_dir, file) 
    pred = model.predict(image_path, confidence=40, overlap=30).json()
    input_point_list = []
    input_label_list = []
    predictor.set_image(img)
    
    if len(pred['predictions'])!=0:
        for j in range(0, len(pred['predictions'])):
            x = int(pred['predictions'][j]['x'])
            y = int(pred['predictions'][j]['y'])
            w = int(pred['predictions'][j]['width'])
            h = int(pred['predictions'][j]['height'])       
            x_c, y_c = calculate_center(x, y, w, h)
            input_point_list.append([int(x_c), int(y_c)])
            input_label_list.append(1)
            # print(input_label_list)
            # x,y,w,h = cv2.boundingRect(cntr)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
        input_point = np.array(input_point_list)
        input_label = np.array(input_label_list)
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        save_path = os.path.join(final_dir, file.split('.')[0] + '_final.jpg')
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(masks[0], plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        plt.show()
        plt.savefig(save_path)
        plt.close('all')
        cv2.imwrite(result_path, img)          
        
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:02:59 2023

@author: bsoltanian
"""

"""

This script is designed for fine-tuning a deep learning model, specifically for
tasks related to solar panel imagery. It utilizes PyTorch for neural network 
operations and OpenCV for image processing, along with other libraries for 
handling and analyzing image data. The main focus is on refining a segmentation
model (indicated by the use of `segment_anything` modules) for enhanced accuracy 
in tasks like defect detection or component segmentation in solar panels. This 
refactoring script includes various utility functions and progress tracking, indicating
 a comprehensive approach to model training and evaluation.
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

from utils import *
from segment_anything.utils.transforms import ResizeLongestSide
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



model_type = 'vit_b'
checkpoint = 'sam_vit_b_01ec64.pth'
device = 'cuda:0' 
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train();

image_paths = 'input_kaggle_original/images'
mask_paths = 'input_kaggle_original/masks'   

list_masks = sorted(Path(mask_paths).iterdir())

for i in range(0, 10):
    print(i)
    bbox_coords = get_bbox_coords(i, list_masks)
    ground_truth_masks = get_ground_truth(bbox_coords, mask_paths)   
    
    transformed_data = defaultdict(dict)
    for k in bbox_coords.keys():
        img_file = '_'.join(k.split('_')[:-1])
        image = cv2.imread(os.path.join(image_paths, '{}.jpg'.format(img_file)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        
        
        transformed_image = image_transform(image, transform, device)
        input_image = sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])
        
        transformed_data[k]['image'] = input_image
        transformed_data[k]['input_size'] = input_size
        transformed_data[k]['original_image_size'] = original_image_size
      
        
    # Set up the optimizer, hyperparameter tuning will improve performance here
    lr = 1e-4
    wd = 0
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
    
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()
    keys = list(bbox_coords.keys())
    
    
    
    num_epochs = 100
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for k in keys: 
            input_image = transformed_data[k]['image'].to(device)
            input_size = transformed_data[k]['input_size']
            original_image_size = transformed_data[k]['original_image_size']
            
            # No grad here as we don't want to optimise the encoders
            with torch.no_grad():
              image_embedding = sam_model.image_encoder(input_image)
              
              prompt_box = bbox_coords[k]
              
              box_torch = bbox_transform(transform, prompt_box, original_image_size, device)
              
              sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                  points=None,
                  boxes=box_torch,
                  masks=None,
              )
            low_res_masks, iou_predictions = sam_model.mask_decoder(
              image_embeddings=image_embedding,
              image_pe=sam_model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=False,
            )
        
            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
        
            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            
            loss = loss_fn(binary_mask, gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        losses.append(epoch_losses)
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
    
    model_save_path = 'model/sam_vit_b_01ec64_{}.pth'.format(i)  # Replace with your desired path
    torch.save(sam_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
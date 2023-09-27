'''
Structure of the Dataset labels -- PASCAL VOC dataset and COCO dataset

class_label of bbox1 in image, x, y, w, h
class_label of bbox2 in image, x, y, w, h
.                               . . . . .
.                               . . . . .
.                               . . . . .
test.csv and train.csv
'''

import config
import numpy as np
import os
import sys
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile

from utils import (
    iou_width_height as iou,
    non_max_suppression as nms,
)

''' We use albumentations '''
ImageFile.LOAD_TRUNCATED_IMAGES = True # to nt get problems while loading images

class YOLODataset(Dataset): # inherit from DataSet
    # for COCO and PASCAL VOC both
    def __init__(self, csv_file, img_dir, label_dir,
                 anchors, image_size=416, S=[13, 26, 512], # for v3 default
                 C=20, # for PASCAL VOC
                 transform=None):
        super(YOLODataset, self).__init__()

        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        # anchors of all the scale prediction, we have 3 anchors for each scale 
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3 # we have 3 scales, so dividing by 3

        self.C = C
        self.ignore_iou_threshold = 0.5

        ''' we set one anchor responsible for each cell in all of the different scale'''
        # for each object a anchor box is responsible, we have to take the box with highest iou
    
    def __len__(self):
        return len(self.annotations)
    
    # labels are relative to the entire image, but in YOLO, everything is relative to the cell
    def __getitem__(self, index):   
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # at the 2nd col -name of the text file
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), shift=4, axis=1).tolist() # ndmin - atleast 2 dimension
        # each row has one bbox, and each row has 5 values - class, x, y, w, h
        # in albumentations, we need [x, y, w, h, class]--- shifting row to the left by 4 positions -- performed along col - axis = 1

        img_path = os.path.join(self.img_dir, self.annotations[index, 0]) # first col is image name
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes) # bbox should be correct even after augmentation
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        # to have same no of anchors for each scale prediction. divide by 3 - scale pred, SxS - no of grids 
        # for each target (# scale pred), we have # anchors, grid size, 6 values - prob of object present + 5 - [p_o, x, y, w, h, c]
        targets = [torch.zeros(self.num_anchors // 3, S, S, 6) for S in self.S] # total 9 anchor boxes, divided by total (3) gives 3.

        for box in bboxes: # each of bbox in the image, which bbox, which cell should be responsible for all scale pred  -  highest IoU
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors) # that box and all of the anchors - width and height
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            # there is an anchor for each of the 3 scale for each of the bbox
            has_anchor = [False, False, False] # init
            # make sures that bbox for each of the scales that are reponsible for predicting the particular object 

            for anchor_idx in anchor_indices: # going throug anchors 
                # which anchor does this belongs to
                scale_idx = anchor_idx // self.num_anchors_per_scale # 0, 1, 2 - which target we have to take out from the combinations
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # bet 0, 1, 2 - dependent on which anchor on that particular scale we are computing 
                S = self.S[scale_idx] # how many cell are in this scale 
                # i -- which y cell
                # j -- which x cell
                i, j = int(S * y), int(S * x)  # x = 0.5, S=13 --> int(6.5) = 6 - 6th cell in the x coordinate, similar to the y coordinate
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # for particular cell
                # 0 - p_o

                if not anchor_taken and not has_anchor[scale_idx]: # if that anchor is not taken and we dont already have an anchor on the scale for this bbox
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1 # assigned if not assigned 
                    # finding the value
                    x_cell =  S*x - j  # 6.5 - 6 = 0.5 - to get the middle of the image in cell
                    y_cell = S*y - i # for the y value of the midpoint # between 0 and 1

                    width_cell, height_cell = ( # relative tot he cell -- greater than 1
                        width * S, # S=13, width=0.5 --> 6.5
                        height * S, 
                    )

                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates # 1-5 - x, y, w, h
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label) # last is class label
                
                # if IoU of bbox > ignore threshold -- make sure that we dont remove that bbox
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore this prediction
                    # we need this 

        return image, tuple(targets)



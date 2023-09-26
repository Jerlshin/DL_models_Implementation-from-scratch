'''PASCAL Visual Object Classification (VOC) dataset'''

# I didn't download the full dataset, i just tested with sample dataset

import torch
import pandas as pd
import os
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
            self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None, # S - split size, B - bounding box, C- no of classes
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
    
    def __len__(self): # length of the annotations
        return len(self.annotations)
    
    # loading the data
    def __getitem__(self, index): # for one example
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # 1st col 
        boxes = [] # for plitting the boxes | coordinates

        with open(label_path) as f:
            for label in f.readlines():

                class_label, x, y, width, height = [  # class - int, and others where float
                    float(x) if float(x) != int(float(x)) else int(x) # if not float, make it float -- it is tring
                    for x in label.replace("/n", "").split()
                ] # reading each line 

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0]) # image 
        image = Image.open(img_path)

        boxes = torch.tensor(boxes) # for making the transforms

        if self.transform:
            image, boxes = self.transform(image, boxes) # send both, as it changes the corrdinates.. so, we need to change the both

        # relative to entrie image
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B)) # s, s, 30 -- multiplying the 5 with no of B, so, it will be 21:25, 25:30 for B=2
        '''but, we only use the 25 nodes, we dont care the other boxes'''
        
        # convert to relative to each cell
        for box in boxes:
            class_label, x, y, width, height = box.tolist() # converting again to list
            class_label = int(class_label) # confirmation
            # S - slit size, no of cell

            i, j = int(self.S * y), int(self.S * x) # y - row, x - col of the cell split 
            # every cell should be 0 or 1
            x_cell, y_cell = self.S * x - j, self.S * y - i # relative to cell. 

            # width, height -- relative to entire image
            width_cell, height_cell = ( # scaling relative to the cell
                width * self.S, 
                height * self.S
            )

            if label_matrix[i, j, 20] == 0: # cel i, j, 20 - p_c # if no object in the cell
                label_matrix[i, j, 20] == 1 # no object
                box_corrdinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                # and settings the corrdinates in the labels
                label_matrix[i, j, 21:25] = box_corrdinates # setting for each cell
                label_matrix[i, j, class_label] = 1 # specifying the class

        return image, label_matrix

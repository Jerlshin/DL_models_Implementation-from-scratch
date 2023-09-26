''' Calculate the Intersection box'''  # To calcualate the Intersection over Union 
# we only have the number 
# Box1 = [x1, y1, x2, y2]
# Box2 = [x1, y1, x2, y2]

import torch

""" Intersection Area parameters
x1 = max(box1[0], box2[0])
y1 = max(box1[1], box2[1])

x2 = min(box1[2], box2[2])
y2 = min(box1[3], box2[3])
"""

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    # boxes_preds.shape - (N, 4)
    # boxes_labels.shape - (N, 4)

    if box_format == "midpoint":
        '''
        # Midpoint = x, y 
        # Width, 
        # Height
        ''' 
        
        # x = boxes_preds[..., 0:1]     # y = boxes_preds[..., 1:2]

        # width = boxes_preds[..., 2:3]     # height = boxes_preds[..., 3:4]

        # Top Left for box 1
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2  
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        # Bottom right for box 1
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        
        # Top Left for box 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2

        # Bottom right for box 2        
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners": 
        '''
        # box_1 (x1, x2, y1, y2 ) 
        # box_2 (x1, x2, y1, y2 )
        '''
        box1_x1 = boxes_preds[..., 0:1] # 0th dim 
        box1_y1 = boxes_preds[..., 1:2] # 1st dim
        box1_x2 = boxes_preds[..., 2:3] # 2nd dim
        box1_y2 = boxes_preds[..., 3:4] # 3rd dim
        # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]  
        box2_y1 = boxes_labels[..., 1:2] 
        box2_x2 = boxes_labels[..., 2:3] 
        box2_y2 = boxes_labels[..., 3:4] 
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)

    x2 = torch.max(box1_x2, box2_x2)
    y2 = torch.max(box1_y2, box2_y2)

    # width = x2 - x1
    # height = y2 - y1

    # if dont intersect, clamp with 0 - intersection should be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    # union = a + b - intersection 
    return intersection / (box1_area + box2_area - intersection + 1e-6) 
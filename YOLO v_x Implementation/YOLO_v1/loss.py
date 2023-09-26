import torch
import torch.nn as nn
from IoU import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20): # 20 classes, 2 bounding box, split size of the image 
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

        self.S = S
        self.B = B
        self.C = C
        
        self.lambda_noobj = 0.5 # low priority, no object in the cell
        self.lambda_coord = 5 # high priority, corrdinator
    
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C, self.B*5) # no of examples is the same (-1)
        # 0-19 - C_i, 20 - p_c, 21-24 - b1_coord, 25-29 - b2_coord  
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # [x, y, w, h]
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) # we have only one target, so it remains the same

        ious = torch.cat([iou_b1.unqueeze(0), iou_b2.unsqueeze(0)], dim=0) # concatenate along dim 0 we just unsqueezed

        iou_maxes, best_box = torch.max(ious, dim=0) # bestbox is argmax ad iou_boxes is the box itself, we dont use this
        # keep the dim, so, we dont slice   # Identity function
        exists_box = target[..., 20].unsqueeze(3) # identity  of object i -- if there is any object in cell i

        ''' ###### For box coordinates ###### '''
        box_predictions = exists_box * (  # only if there is object  
            best_box * predictions[..., 26:30] # if the 2nd bbox is correct
            + (1 - best_box) * predictions[..., 21:25] # bestbox will be 0 if the first bbox is perfect
        )

        box_targets = exists_box * target[..., 1:25] # target hs only one box

        # sqrt of width and height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt( # the sign should remain the same, as we removed the sign 'abs' for taking sqrt
            torch.abs(box_predictions[..., 2:4]) + 1e-6 # for numerical instability   # we no need neg value
            )
        # (N, S, S, 25) - target
        box_targets[...,2:4] = torch.sqrt(box_targets[..., 2:4])
        # (N, S, S, 4) -> (N*S*S, 4) # 4 for bounding boxes
        box_loss= self.mse(
            torch.flatten(box_predictions, end_dim=-2), # -2 flatten everything before 
            torch.flatten(box_loss, end_dim=-2)
        )

        ''' ###### For Object loss ###### '''
        
        pred_box = ( # if there actually exits a block - 
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )
        # (N*S*S) - everything will be flatten
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        ''' ###### For No-Object loss ###### '''
        # (N, S, S, 1) -> (N, S*S)
        # for BOX1
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )
        # For BOX2
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse

        
        ''' ###### For class loss ###### '''

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., 20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        '''TOTAL LOSS'''

        loss = {
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        }

        return loss
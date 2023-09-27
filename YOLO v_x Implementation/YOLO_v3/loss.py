import torch
import torch.nn as nn

from utils import intersection_over_union 


class YoloLoss():
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss() # for box prediction
        self.bce = nn.BCEWithLogitsLoss() # aplied with sigmoid
        self.entropy = nn.CrossEntropyLoss() # as we have only one class per box - we dont have multi label class
        self.sigmoid = nn.Sigmoid() 
    
        # constants
        '''how much each loss should have effect on the total loss'''
        self.lambda_class = 1
        self.lambda_noobj = 10 # we value no obj more
        self.lambda_obj = 1
        self.lambda_box = 10
    
    def forward(self, predictions, target, anchors): # for particular scale. 
        # loss for each different scales --- prev, we ignored IoU greated than 0.5
        # binary tensor
        obj = target[..., 0] == 1 # if there is a object
        noobj = target[..., 0] == 0 # if no object 

        '''No object loss'''
        no_object_loss = self.bce( # full slices before the last one     
            (predictions[...,0:1][noobj], (target[..., 0:1][noobj])), # we do 0:1 to prevent the batch dimentsion 
        )

        '''Object loss'''
        # 3x2 # 3 anchors per scale, and height and width
        # to multiply will all anchors and all cells in the anchors, we add the extra dim  -- brodcasting 
        anchors = anchors.reshape[1, 3, 1, 1, 2] # match the dim of weight and height we multiply - p_w * exp(t_w)
        box_pred = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_pred[obj], target[..., 1:5][obj]).detach() # no need grads

        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1]))

        '''Box coordinate loss'''
        # change the target instead of the predictions
        predictions[...,1:3] = self.sigmoid(predictions[...,1:3]) # x, y between 0 and 1
        # opposite of the exponent in the object loss
        #### To have better gradient flow
        target[..., 3:5] = torch.log( # width and height
            (1e-6 + target[..., 3:5] / anchors)
        )        
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        '''Class loss'''
        class_loss = self.entropy(
            (predictions[...,5:][obj]), (target[..., 5][obj].long())
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
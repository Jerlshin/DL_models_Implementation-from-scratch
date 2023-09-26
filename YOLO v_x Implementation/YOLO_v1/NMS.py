import torch

from IoU import intersection_over_union

'''
while BoundingBoxes:
    - Take out the largest prob box
    - Remove all other boxes with IoU > threshold
'''
def non_max_suppression(bboxes, iou_threshold, 
                        prob_threshold, box_format="corners"):
    
    # predictions = [[class, prob, x1,, y1, x2, y2]] - 

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold] # remove the box with low prob
    bboxes_after_nms = []

    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) # high -> low   sorting 

    while bboxes: # while we have bounding box
        # choose 1 box
        chosen_box = bboxes.pop(0) # taking the bbox with largest prob
        
        bboxes = [ # compare the chosen box to the other boxes 
            box 
            for box in bboxes # compaing to the other boxes
            if box[0] != chosen_box[0] # we dont want to compare the same box
            or intersection_over_union(
                torch.tensor(chosen_box[2:]), # remove the 1st and 2nd element
                torch.tensor(box[2:]),
                box_format=box_format,
            ) 
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms 


     
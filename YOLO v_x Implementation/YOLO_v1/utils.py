import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # for amkign retangleshapes 
from collections import Counter

from IoU import intersection_over_union
from NMS import non_max_suppression
from mAP import mean_avg_precision

def plot_image(image, boxes): # plotting the predicted bbox on the image
    im = np.array(image)
    height, width, _ = im.shape # no need channel
    # x, y, w, h

    fig, ax = plt.subplots(1)
    ax.imshow(im)
    
    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        # logic as in IoU module  
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(loader, model, iou_threshold,
               threshold, pred_format="cells",
               box_format="midpoint",
               device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make it in eval mode
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels) # true box
        bboxes = cellboxes_to_boxes(predictions) # predicted box

        for idx in range(batch_size):
            nms_boxes = non_max_suppression( # getting only best bboxes
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            '''no need for now'''
            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



'''Converting bboxes output from model with S into entire image rations rather than relative to cell ratios'''
def convert_cellboxes(predictions, S=7): # vectorized implementations

    # pre processing
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    # predicted  are in the shape of (batch_size, 7, 7, 2, 15) - each cell 7x7 grid for two bbox
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    
    # getting the 2 boxes
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    
    scores = torch.cat( # concat both bbox with dim 0 - created
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    # take the best box. and then remove the added dim 
    best_box = scores.argmax(0).unsqueeze(-1) # 0 - batch_dimension 
    
    # removing the low priority box and keeping only the best box 
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2 
    # represents the indices of the cells in the grid. 
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1) # adding the new dima nd the end 
    # box coordinates 
    # converting the relative coordinates of the bbox to abosolute image coordinates using the formula - notebook
    '''
    x = (1/S) * (relative_x + cell_index_x)
    y = (1/S) * (relative_y + cell_index_y)
    w_y = width and height scaled by 1/S
    '''
    x = 1 / S * (best_boxes[..., :1] + cell_indices) # x
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3)) 
    w_y = 1 / S * best_boxes[..., 2:4]
    
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )

    # concatenating all the results along the last axis
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

'''saving the model at each checkpoint'''
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

'''Mean Average Precision - metric to evaluate the Object detection model - mAP'''
import torch

'''
1. Get all bounding box predictions on out test set
2. Sort by decending confidence score
3. Calculate the Precision and Recall as we go through all outputs
4. Plot the Precision(y)-Recall(x) graph
5. Calculate the Area under PR curve
6. Calculate this for every class
7. Take the average -- mAP
'''

from collections import Counter
from IoU import  intersection_over_union

def mean_avg_precision(
        pred_boxes,  # list of all bbox pred [ [train_idx, class_pred, prob_score, x1, x2, y1, y2], ...]
        true_boxes, 
        iou_threshold=0.5, box_format='corners',
        num_classes=20 
):
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths= []

        for detection in pred_boxes:
            if detection[1] == c: # if class prediction of the box is equal to the current iter class c
                detection.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # img0 has 3 bbox
        # img1 has 5 bbox
        '''creates a dict'''
        amount_bboxes = Counter([gt[0] for gt in ground_truths]) # amount_bboxes = {0:3, 1:5} 

        for key, val in amount_bboxes.items():
            amount_bboxes[key] == torch.zeros(val) # instead of 3, we have 000 - 3 zeroes -- assigning to value of the key

        # amount_boxes = {0:torch.tensor([0, 0, 0]), 1:torch.tensor([0, 0, 0, 0, 0])}
        detections.sort(key=lambda x: x[2], reverse=True) # descending order of prob score

        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))

        total_true_bboxes = len(ground_truths)
        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


'''
Classification Metrics:
Accuracy = Total correct / Total observation

Precision = TP / TP + FP 

Recall = TP / TP + FN
'''


# True Positive ----> For a target bouding box - we predicted the box with IoU > threshold  
# False Positive ----> For a target bounding box - IoU is less than a threshold

# False Negatives ----> For a existing target, it didn't predict a bbox
# False Positive ----> Didn't output a bbox, for non existing bbox =====> Not considered in out case
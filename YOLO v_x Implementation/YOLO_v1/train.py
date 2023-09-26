from typing import Any
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader

# utilities we need
from model import Yolov1
from dataset import VOCDataset
from utils import plot_image, get_bboxes, cellboxes_to_boxes, save_checkpoint, load_checkpoint

# loading the metrics
from IoU import intersection_over_union
from NMS import non_max_suppression
from mAP import mean_avg_precision

# loss function - paper
from loss import YoloLoss

import time
import sys
import os


seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 2e-5  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16 # 64 in the paper -- RIP laptop

'''we should change when actual training'''
WEIGHT_DECAY = 0  # it will train for long. 
EPOCHS = 100 # change
NUM_WORKERS = 2 # parallel
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "model_file.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

IOU_THRESHOLD = 0.5

# for image transformation ... we cant use in built transforms as we have bbox

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, bboxes):
        for t in self.transforms: # operate only on image
            img, bboxes = t(img), bboxes
        
        return img, bboxes

transform = Compose(
    [
        transforms.Resize((448, 448)), 
        transforms.ToTensor()
    ]
)

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        X, y = x.to(DEVICE), y.to(DEVICE)
        out = model(X)
        loss = loss_fn(out, y) 
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss: {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(split_size = 7, num_boxes=2, 
                   num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    train_dataset = VOCDataset(
        "data/8examples.csv",
        transform=transform,
        img_dir = IMG_DIR,
        label_dir = LABEL_DIR  
    )

    # we dont give the transform for test data, but for training purpose, we give
    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir = IMG_DIR, label_dir = LABEL_DIR  
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True # if last batch has only few examples, that is unneccasry
    )

    train_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY, shuffle=True, drop_last=True 
    ) 

    ''' ----TESTING------''' # comment this while training 
    for epoch in range(EPOCHS):
        for x, y in train_loader:
            x = x.to(DEVICE)
            for idx in range(8):
                bboxes = cellboxes_to_boxes(model(x))
                bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
                plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
            
            sys.exit()
    # set load_model to true fro test

        '''===TRAINING==='''
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=IOU_THRESHOLD, threshold=0.4
        )

        mean_avg_prec = mean_avg_precision(
            pred_boxes, target_boxes, iou_threshold=IOU_THRESHOLD, box_format="midpoint" # we can also give "center"
        )

        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_loader, model, optimizer, loss_fn)
        
        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(10)

if __name__ == "__main___":
    main()
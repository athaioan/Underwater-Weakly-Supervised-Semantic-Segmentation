import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import importlib
from types import SimpleNamespace
from utils import *
import torch.nn as nn
import pandas as pd
# from models.modeling import VisionTransformer, CONFIGS
from networks import VGG16, VGG16_multiview
from focal_loss import FocalLoss

### Setting arguments
args = SimpleNamespace(epochs=6,
                       batch_size=1,
                       lr=1e-4,
                       weight_decay=5e-4,
                       input_dim=448,
                       pretrained_weights="pretrained/vgg16_20M.pth",
                       img_folder="MMT-Datasetv3/triplets/",
                       train_set="train_multiview.txt",
                       val_set="val_multiview.txt",
                       test_set="test_multiview.txt",
                       labels_dict="class_dict.npy",
                       fl=False,
                       frozen_stages = [1],  ## freezing the very generic early convolutional layers
)

# #### Stage 1
args.session_name = "multiview/"
#


## Constructing the training loader
train_loader = MMTDataset_multiview(args.train_set, args.labels_dict, args.img_folder, args.input_dim)
train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True)

## Constructing the validation loader
val_loader = MMTDataset_multiview(args.val_set, args.labels_dict, args.img_folder, args.input_dim)
val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False)

## Constructing the test loader
test_loader = MMTDataset_multiview(args.test_set, args.labels_dict, args.img_folder, args.input_dim)
test_loader = DataLoader(test_loader, batch_size=args.batch_size, shuffle=False)

## Initializing the model
model = VGG16_multiview(n_classes=18).cuda()
model.epochs = args.epochs
model.session_name = args.session_name
model.load_pretrained(args.pretrained_weights)
model.freeze_layers(args.frozen_stages)

if not os.path.exists(model.session_name):
    os.makedirs(model.session_name)

param_groups = model.get_parameter_groups()
optimizer = PolyOptimizer([
    {'params': param_groups[0], 'lr': 8*args.lr, 'weight_decay': args.weight_decay},
    {'params': param_groups[1], 'lr': 16 * args.lr, 'weight_decay': 0},
    {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.weight_decay},
    {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0}
], lr=args.lr, weight_decay=args.weight_decay, max_step=len(train_loader)*args.epochs)


if args.fl:
    ## focal loss
    criterion = FocalLoss()
else:
    ## frequency-based weighting
    class_weight = 45117 / torch.tensor([30160,	2,	9757,	1004,	4,	205,	833,	4,	252,	21,	7,	1366,	1323,	83,	56,	26,	1,	13]).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weight)


for current_epoch in range(model.epochs):

    model.epoch = current_epoch

    print("Training epoch...")
    model.train_epoch(train_loader, optimizer,criterion)

    print("Validating epoch...")
    conf_matrix = model.val_epoch(val_loader,criterion)
    model.visualize_graph()

    if model.val_history["loss"][-1] < model.min_val:
        print("Saving model...")
        model.min_val = model.val_history["loss"][-1]

        torch.save(model.state_dict(), model.session_name+"stage_1.pth")
        df = pd.DataFrame(conf_matrix)
        filepath = args.session_name+'val_conf_matrix.xlsx'
        df.to_excel(filepath, index=False)


## Initializing the model
model = VGG16_multiview(n_classes=18).cuda()
model.epochs = args.epochs
model.session_name = args.session_name
model.load_pretrained(model.session_name+"stage_1.pth")


model.cuda()

#
#
#
#
# print("testing epoch...")
# model.epoch = 0
# model.epochs = 0
#
# conf_matrix = model.val_epoch(test_loader, criterion)
# df = pd.DataFrame(conf_matrix)
# filepath = 'test_conf_matrix.xlsx'
# df.to_excel(filepath, index=False)
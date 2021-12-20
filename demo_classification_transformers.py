import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import importlib
from types import SimpleNamespace
# from utils_transformer import *
from utils import *
import torch.nn as nn
import pandas as pd
from models.modeling import VisionTransformer, CONFIGS

### Setting arguments
args = SimpleNamespace(epochs=14,
                       batch_size=4,
                       lr=0.5e-3,
                       weight_decay=5e-3,
                       input_dim=448,
                       pretrained_weights="pretrained_ViT-B_16.pth",
                       img_folder="MMT-Datasetv3/triplets/",
                       train_set="train_new.txt",
                       val_set="val_new.txt",
                       test_set="test_new.txt",
                       step_update_lr=4,

                       # train_set="train_trash.txt",
                       # val_set="val_trash.txt",
                       labels_dict="class_dict.npy",
                       frozen_stages=[1], ## freezing the very generic early convolutional layers
                       )

# #### Stage 1
args.session_name = "transformerssda/"
#

loader_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

## Constructing the training loader
train_loader = MMTDataset_baseline(args.train_set, args.labels_dict, args.img_folder, args.input_dim, transformer = True)


train_loader_T = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True)
train_loader = iter(train_loader_T)

## Constructing the validation loader
val_loader = MMTDataset_baseline(args.train_set, args.labels_dict, args.img_folder, args.input_dim, transformer = True)
val_loader = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False) ## no point in shufflying the validation data

## Initializing the model
config = CONFIGS["ViT-B_16"]
num_classes = 18
model = VisionTransformer(config, args.input_dim, zero_head=True, num_classes=num_classes)

model.epochs = args.epochs
model.session_name = args.session_name
model.load_pretrained(args.pretrained_weights)
model.cuda()

if not os.path.exists(model.session_name):
    os.makedirs(model.session_name)

# Prepare optimizer and scheduler
optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=0.9,
                            weight_decay=args.weight_decay)


class_weight = 45117 / torch.tensor([30160,	2,	9757,	1004,	4,	205,	833,	4,	252,	21,	7,	1366,	1323,	83,	56,	26,	1,	13]).cuda()

criterion = nn.CrossEntropyLoss(weight=class_weight)


for current_epoch in range(model.epochs):

    if current_epoch % args.step_update_lr == args.step_update_lr-1:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']/3

    model.epoch = current_epoch

    print("Training epoch...")
    model.train_epoch(train_loader_T,train_loader, optimizer,criterion)

    print("Validating epoch...")
    conf_matrix = model.val_epoch(val_loader,criterion)
    model.visualize_graph()

    if model.val_history["loss"][-1] < model.min_val:
        print("Saving model...")
        model.min_val = model.val_history["loss"][-1]

        torch.save(model.state_dict(), model.session_name+"stage_1.pth")
        df = pd.DataFrame(conf_matrix)
        filepath = 'val_conf_matrix.xlsx'
        df.to_excel(filepath, index=False)


class_weight = 45117 / torch.tensor([30160,	2,	9757,	1004,	4,	205,	833,	4,	252,	21,	7,	1366,	1323,	83,	56,	26,	1,	13]).cuda()

criterion = nn.CrossEntropyLoss(weight=class_weight)
## Constructing the validation loader
test_loader = MMTDataset_baseline(args.train_set, args.labels_dict, args.img_folder, args.input_dim, transformer = True)

test_loader = DataLoader(test_loader, batch_size=args.batch_size, shuffle=False) ## no point in shufflying the validation data
## Initializing the model
config = CONFIGS["ViT-B_16"]
num_classes = 18
model = VisionTransformer(config, args.input_dim, zero_head=True, num_classes=num_classes)

model.session_name = args.session_name
model.load_pretrained(model.session_name+"stage_1.pth")
model.cuda()
print("testing epoch...")
model.epoch = 0
model.epochs = 0

conf_matrix = model.val_epoch(test_loader, criterion)
df = pd.DataFrame(conf_matrix)
filepath = 'test_conf_matrix.xlsx'
df.to_excel(filepath, index=False)
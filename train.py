import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from argparse import ArgumentParser
import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange

from vit_pytorch import LongVViT
from utils.data import UCF101, SMTHV2
from torchvision import transforms

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import preprocess

# Parse arguments
parser = ArgumentParser()

parser.add_argument("--annotations", type=str, default="dataset/ucf/trainlist03.txt", help="Dataset labels path")
parser.add_argument("--root-dir", type=str, default="dataset/ucf/frames", help="Dataset files root-dir")
parser.add_argument("--classes", type=int, default=101, help="Number of classes")
parser.add_argument("--frames", type=int, default=16, help="Number of frames")

parser.add_argument("--dataset", choices=['ucf', 'smth'], default='ucf')
parser.add_argument("--weight-path", type=str, default="weights/ucf/v1", help='Path to save weights')
parser.add_argument("--resume", type=int, default=0, help='Resume training from')

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay")
parser.add_argument("--epochs", type=int, default=22, help="Number of epochs")
parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split")

# Learning scheduler
LRS = [1, 0.1, 0.01]
STEPS = [1, 15, 19]

# Parse arguments
args = parser.parse_args()
print(args)

# Load model
model = LongVViT(image_size=224, patch_size=32, dim=1024, num_classes=args.classes, depth=3, heads=8, mlp_dim=128, attention_window=7, 
                frames=frames, attention_mode='sliding_chunks', dropout=0.1, emb_dropout=0.1)

if torch.cuda.is_available():
    model = nn.DataParallel(model).cuda()

# Resume weights
if args.resume > 0:
  model.load_state_dict(torch.load(f'{args.weight_path}/weights_{args.resume}.pth'))   

# Load dataset
if args.dataset == 'ucf':
  dataset = UCF101(args.annotations, args.root_dir, preprocess=resnet_preprocess, classes=args.classes, frames=args.frames)
elif args.dataset == 'smth':
  dataset = SMTHV2(args.annotations, args.root_dir, preprocess=resnet_preprocess, frames=args.frames)

# Split
train_set, val_set = random_split(dataset, [len(dataset) - int(args.validation_split * len(dataset)), int(args.validation_split * len(dataset))] )
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=8, persistent_workers=True)


# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

softmax = nn.LogSoftmax(dim=1)

def adjust_learning_rate(optimizer, epoch):

    """Sets the learning rate to the according to POLICY"""
    for ind, step in enumerate(STEPS):
      if epoch < step:
        break
    ind = ind - 1

    lr = args.learning_rate * LRS[ind]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



for epoch in range(max(args.resume+1, 1), args.epochs+1):
    
    # Adjust learning rate
    adjust_learning_rate(optimizer, epoch)

    progress = tqdm(train_loader, desc=f"Epoch: {epoch}, loss: 0.000")
    for src, target in progress:
        
        # print(src.shape, target.shape)
        # src, target = train_loader[i] 
        if torch.cuda.is_available():
            src = torch.autograd.Variable(src).cuda()
            target = torch.autograd.Variable(target).cuda()
        optimizer.zero_grad()
        # Forward + backprop + optimize
        output = model(src)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        # Show loss
        loss_val = loss.item()
        progress.set_description(f"Epoch: {epoch}, loss: {loss_val}")

    # Validation
    val_loss = 0
    val_acc = 0
    for src, target in tqdm(val_loader, desc=f"Epoch: {epoch}, validating"):
        # src, target = train_loader[i]
        if torch.cuda.is_available():
            src = torch.autograd.Variable(src).cuda()
            target = torch.autograd.Variable(target).cuda()
        
        with torch.no_grad():
            output = model(src)
            loss = loss_func(output, target)
            val_loss += loss.item()
            output = softmax(output)
            
            val_acc += torch.sum(torch.argmax(output, dim=1) == target).cpu().detach().item() / args.batch_size

    print("Validating loss:", val_loss/len(val_loader), ", accuracy:", val_acc/len(val_loader))

    # Save weights
    torch.save(model.state_dict(), f'{args.weight_path}/weights_{epoch}.pth')

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from argparse import ArgumentParser, Namespace
import torch
torch.backends.cudnn.benchmark = True
import yaml

import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange

from vit_pytorch import LongViT
from utils.data import ImageNet
from torchvision import transforms

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utils import preprocess

# Parse arguments
parser = ArgumentParser()

parser.add_argument("--annotations", type=str, default="dataset/kinetics/annotations.json", help="Dataset labels path")
parser.add_argument("--root-dir", type=str, default="dataset/kinetics/train", help="Dataset files root-dir")
parser.add_argument("--val-annotations", type=str, default="dataset/kinetics/val.json", help="Validation labels")
parser.add_argument("--val-root-dir", type=str, default="dataset/kinetics/val", help="Dataset files root-dir")
parser.add_argument("--classes", type=int, default=1000, help="Number of classes")
parser.add_argument("--config", type=str, default='configs/longViT.yaml', help="Config file")

parser.add_argument("--dataset", choices=['imagenet'], default='imagenet')
parser.add_argument("--weight-path", type=str, default="weights/imagenet/v1", help='Path to save weights')
parser.add_argument("--resume", type=int, default=0, help='Resume training from')

# Hyperparameters
parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

# Learning scheduler
LRS = [1, 0.1, 0.01]
STEPS = [1, 15, 19]

# Parse arguments
args = parser.parse_args()
print(args)

# Load config
with open(args.config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cfg = Namespace(**cfg)

# Load model
model = LongViT(**vars(cfg))

if torch.cuda.is_available():
    model = nn.DataParallel(model).cuda()

# Resume weights
if args.resume > 0:
  model.load_state_dict(torch.load(f'{args.weight_path}/weights_{args.resume}.pth'))   

# Load dataset
if args.dataset == 'imagenet':
  train_set = ImageNet(args.annotations, args.root_dir, preprocess=preprocess, train=True)
  val_set = ImageNet(args.val_annotations, args.val_root_dir, preprocess=preprocess, train=False)

# Split
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=16, persistent_workers=True)


# Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

softmax = nn.LogSoftmax(dim=1)

# def adjust_learning_rate(optimizer, epoch):

#     """Sets the learning rate to the according to POLICY"""
#     for ind, step in enumerate(STEPS):
#       if epoch < step:
#         break
#     ind = ind - 1

#     lr = args.learning_rate * LRS[ind]
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr



for epoch in range(max(args.resume+1, 1), args.epochs+1):
    
    # # Adjust learning rate
    # adjust_learning_rate(optimizer, epoch)

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

        # Summary
        if i % 100 == 99:
          tensorboard.add_scalar('train_loss', loss_val, epoch * len(train_loader) + i)

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

    # Summary
    tensorboard.add_scalar('val_loss', val_loss/len(val_loader), epoch)
    tensorboard.add_scalar('val_acc', val_acc/len(val_loader) * 100, epoch)

    # Save weights
    torch.save(model.state_dict(), f'{args.weight_path}/weights_{epoch}.pth')

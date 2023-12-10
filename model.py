import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from random import shuffle, seed

import torch
from torch.utils.data import Dataset

import config
from utils import compute_frames
import torch.optim as optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import FocalLoss

# set device to GPU
dev = "cuda:0"
f_loss = FocalLoss("binary", gamma=5).to(dev)


def init_model(in_channels):
    # set random seeds for reproductibility
    torch.manual_seed(123)
    seed(123)
    np.random.seed(123)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels, 
        classes=2,
    ).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.epoch = 0
    model.errs = []
    return model, optimizer

def evaluate_model(model, dataloader):
    err = 0
    for inputs, labels in dataloader:
        y_pred = model(inputs).detach()
        # err += ce_loss(input=y_pred, target=labels)
        err += f_loss(y_pred=y_pred, y_true=labels)
    err = err / len(dataloader)

    return err

def run_epoch(model, optimizer, trainloader):
    model.epoch += 1
    print(f"\nEpoch {model.epoch}")
    
    train_err = 0
    for inputs, labels in tqdm(trainloader):
        y_pred = model(inputs)
        # loss = ce_loss(input=y_pred, target=labels)
        loss = f_loss(y_pred=y_pred, y_true=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_err += loss.detach()
    train_err = train_err / len(trainloader)
    
    return train_err


def train(model, optimizer, n_epochs, trainloader, valloader):
    for epoch in range(n_epochs):
        
        # train for 1 epoch and compute error
        train_err = run_epoch(model, optimizer, trainloader)

        # compute validation error and save history
        train_err = evaluate_model(model, trainloader)
        val_err = evaluate_model(model, valloader)
        model.errs.append([train_err, val_err])

        print(f"Epoch {model.epoch}: Train Loss = {train_err:.6f} | Validation Loss = {val_err:.6f}")


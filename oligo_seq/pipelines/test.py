import argparse
from typing import Any
import yaml
import sys
import os
import json
from tqdm import tqdm
import time
from datetime import datetime
import logging
import copy
sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.utils import data
from torch import optim
import optuna
from oligo_seq.database import *
from oligo_seq.models import *
import wandb

def eval_epoch(model: nn.Module, dataloader: data.DataLoader, loss: nn.Module, device: torch.device) -> float:
    model.eval()
    cumulative_loss = torch.zeros(1,).to(device)
    with torch.no_grad():
        for batch in dataloader:
            batch_device = []
            for t in batch:
                batch_device.append(t.to(device))
            data = batch_device[:-1]
            label = batch_device[-1]
            pred = model(*data)
            cumulative_loss += loss(pred, label)
    loss = cumulative_loss/len(dataloader)
    print(len(dataloader))
    return loss.item()

def main():
    # Evaluate the result obtained on teh server

    with open("data/models/other/lstm_0.json") as f:
        h_par = json.load(f)
    model = OligoLSTM(**h_par["model"])
    # model = OligoMLP(**h_par["model"])
    model.load_state_dict(torch.load("data/models/other/lstm_0.pt", map_location=torch.device('cpu')))

    # dataset, model, optimizer initialization
    batch_size = 128
    dataset = RNNDataset(path="data/datasets/artificial_dataset_35_35.csv")
    # dataset = MLPDataset(path="data/datasets/artificial_dataset_35_35.csv")
    generator = torch.Generator().manual_seed(1234)
    split_lenghs = [round(len(dataset)*length)  for length in [0.4, 0.2, 0.4]]
    split_lenghs[-1] = len(dataset) - sum(split_lenghs[:-1]) # adjust in case there are rounding errors
    train, validation, test = data.random_split(dataset=dataset, lengths=split_lenghs, generator=generator)
    # collate_fn = None
    collate_fn = pack_collate
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = torch.utils.data.DataLoader(dataset=validation, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    loss = torch.nn.MSELoss()

    val_loss = eval_epoch(model=model, dataloader=validation_loader, loss=loss, device='cpu')
    print(f"validation loss : {val_loss}")

if __name__ == "__main__":
    main()
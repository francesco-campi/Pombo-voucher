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
    cumulative_loss = torch.zeros(1, dtype=torch.float64).to(device)
    predictions = None
    with torch.no_grad():
        for batch in dataloader:
            batch_device = []
            for t in batch:
                batch_device.append(t.to(device))
            data = batch_device[:-1]
            label = batch_device[-1]
            pred = model(*data)
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat([predictions, pred])
            l = loss(pred, label)
            cumulative_loss += l
    loss = cumulative_loss/len(dataloader)
    print("Predictions")
    print(predictions)
    return loss.item()

def main():
    # Evaluate the result obtained on teh server
    device = torch.device("cuda") if torch.cuda.is_available() is True else torch.device("cpu")
    with open("data/models/lstm/lstm_0.json") as f:
        h_par = json.load(f)
    model = OligoLSTM(**h_par["model"])
    # # model = OligoMLP(**h_par["model"])
    model.load_state_dict(torch.load("data/models/other/lstm_0.pt", map_location=torch.device(device=device)))
    model = model.to(device)

    # dataset, model, optimizer initialization
    batch_size = 2
    dataset = RNNDataset(path="data/datasets/artificial_dataset_35_35.csv")
    # collate_fn = None
    collate_fn = pack_collate
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    # loader2 = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    # loader3 = torch.utils.data.DataLoader(dataset=dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    loss = torch.nn.MSELoss()
    val_loss = eval_epoch(model=model, dataloader=loader, loss=loss, device=device)
    print(f"validation loss : {val_loss}")
    # val_loss = eval_epoch(model=model, dataloader=loader2, loss=loss, device=device)
    # print(f"validation loss (2): {val_loss}")
    # val_loss = eval_epoch(model=model, dataloader=loader3, loss=loss, device=device)
    # print(f"validation loss (2): {val_loss}")


if __name__ == "__main__":
    main()
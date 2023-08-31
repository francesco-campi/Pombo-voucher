import argparse
from typing import Any
import yaml
import sys
import os
import json
from tqdm import tqdm
import time
sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.utils import data
from torch import optim
import optuna
from oligo_seq.database import *
from oligo_seq.models import *





class Objective:

    def __init__(self, config, dataset) -> None:
        self.config = config
        self.dataset = dataset

    def __call__(self, trail: optuna.Trial) -> Any:

        ################
        # define model #
        ################

        hyperparameters = {}
        hyperparameters["dataset"] = {"mean": self.dataset.mean.item(), "std": self.dataset.std.item()}
        hyperparameters["model"] = {}
        if self.config["model"] == "mlp":
            hyperparameters["model"]["input_size"] = self.config["input_size"]
            hyperparameters["model"]["hidden_size"] = trail.suggest_int("hidden_size", low=self.config["hidden_size"][0], high=self.config["hidden_size"][1])
            hyperparameters["model"]["n_layers"] = trail.suggest_int("n_layers", low=self.config["n_layers"][0], high=self.config["n_layers"][1])
            hyperparameters["model"]["act_function"] = trail.suggest_categorical("act_function", choices=self.config["act_function"])
            hyperparameters["model"]["dropout"] = trail.suggest_float("dropout", low=self.config["dropout"][0], high=self.config["dropout"][1])
            model = OligoMLP(**hyperparameters["model"])
            collate_fn = None
        elif self.config["model"] == "rnn":
            hyperparameters["model"]["input_size"] = self.config["input_size"]
            hyperparameters["model"]["features_size"] = self.config["features_size"]            
            hyperparameters["model"]["hidden_size"] = trail.suggest_int("hidden_size", low=self.config["hidden_size"][0], high=self.config["hidden_size"][1])
            hyperparameters["model"]["n_layers"] = trail.suggest_int("n_layers", low=self.config["n_layers"][0], high=self.config["n_layers"][1])
            hyperparameters["model"]["n_layers_mlp"] = trail.suggest_int("n_layers_mlp", low=self.config["n_layers_mlp"][0], high=self.config["n_layers_mlp"][1])
            hyperparameters["model"]["act_function"] = trail.suggest_categorical("act_function", choices=self.config["act_function"])
            hyperparameters["model"]["nonlinearity"] = trail.suggest_categorical("nonlinearity", choices=self.config["nonlinearity"])
            hyperparameters["model"]["pool"] = trail.suggest_categorical("pool", choices=self.config["pool"])            
            hyperparameters["model"]["dropout"] = trail.suggest_float("dropout", low=self.config["dropout"][0], high=self.config["dropout"][1])
            model = OligoRNN(**hyperparameters["model"])
            collate_fn = pack_collate
        elif self.config["model"] == "lstm":
            hyperparameters["model"]["input_size"] = self.config["input_size"]
            hyperparameters["model"]["features_size"] = self.config["features_size"]
            hyperparameters["model"]["hidden_size"] = trail.suggest_int("hidden_size", low=self.config["hidden_size"][0], high=self.config["hidden_size"][1])
            hyperparameters["model"]["n_layers"] = trail.suggest_int("n_layers", low=self.config["n_layers"][0], high=self.config["n_layers"][1])
            hyperparameters["model"]["act_function"] = trail.suggest_categorical("act_function", choices=self.config["act_function"])
            hyperparameters["model"]["pool"] = trail.suggest_categorical("pool", choices=self.config["pool"])            
            hyperparameters["model"]["dropout"] = trail.suggest_float("dropout", low=self.config["dropout"][0], high=self.config["dropout"][1])
            model = OligoLSTM(**hyperparameters["model"])
            collate_fn = pack_collate
        else:
            raise ValueError(f"{self.config['model']} is not supported")
        
        #####################
        # define dataloader #
        #####################

        generator = torch.Generator().manual_seed(self.config["split_seed"])
        train, validation, test = data.random_split(dataset=self.dataset, lengths=self.config["split_lengths"], generator=generator)
        batch_size = trail.suggest_int("batch_size", low=self.config["batch_size"][0], high=self.config["batch_size"][1])
        train_loader = data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        validation_loader = data.DataLoader(dataset=validation, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        ####################
        # define optimizer #
        ####################

        lr = trail.suggest_float("lr", low=self.config["lr"][0], high=self.config["lr"][1], log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=self.config["scheduler_factor"], patience=self.config["scheduler_patience"])
        loss = nn.MSELoss()

        ###################
        # train the model #
        ###################

        max_patience = self.config["patience"] # for early sotpping
        best_validation_loss = None
        best_model = model.state_dict()
        patience = 0
        epoch_best_model = 0
        for i in range(self.config["n_epochs"]):
            start = time.time()
            train_loss = self.train_epoch(model=model, dataloader=train_loader, loss=loss, optimizer=optimizer)
            validation_loss = self.eval_epoch(model=model, dataloader=validation_loader, loss=loss)
            # wandb log
            scheduler.step(validation_loss)
            if best_validation_loss is None or validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                # patience reset 
                best_model = model.state_dict()
                patience = 0
            else:
                # patience update
                patience += 1
            if patience >= max_patience:
                #restore teh best model
                model.load_state_dict(best_model)
                break
            if i % 5 == 0:
                print(f"Epoch {i+1}: \t Train Loss: {train_loss}, \t Validation Loss: {validation_loss}, Computation time : {time.time() - start}")

        ###################
        # store the model #
        ###################

        model_dir = os.path.join(self.config["models_path"], self.config["model"])
        os.makedirs(model_dir, exist_ok=True)
        model_file = f"{self.config['model']}_{trail.number}.pt"
        torch.save(model.state_dict(), os.path.join(model_dir, model_file))
        hyperparameters_file = f"{self.config['model']}_{trail.number}.json"
        with open(os.path.join(model_dir, hyperparameters_file), 'w') as f:
            json.dump(hyperparameters, f)

        return best_validation_loss


    def train_epoch(self, model: nn.Module, dataloader: data.DataLoader, loss: nn.Module, optimizer: optim.Optimizer) -> float:
        model.train()
        cumulative_loss = 0.
        for data, label in dataloader:
            pred = model(data)
            batch_loss = loss(pred, label)
            cumulative_loss =+ batch_loss
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss = cumulative_loss/len(dataloader)
        return loss.item()


    def eval_epoch(self,model: nn.Module, dataloader: data.DataLoader, loss: nn.Module) -> float:
        model.eval()
        cumulative_loss = 0.
        with torch.no_grad():
            for data, label in dataloader:
                pred = model(data)
                cumulative_loss += loss(pred, label)
        loss = cumulative_loss/len(dataloader)
        return loss.item()


def main():

    #########################
    # read input arguments #
    #########################

    parser = argparse.ArgumentParser(
        prog="Artificial Dataset",
        usage="generate_artificial_dataset [options]",
        description=main.__doc__,
    )
    parser.add_argument("-c", "--config", help="path to the configuration file", default="config/generate_artificial_dataset.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    ##################
    # define dataset #
    ##################

    # we define the dataset outside the trial function to avoid processig the data multiple times
    if config["model"] == "mlp":
        dataset = MLPDataset(path=config["dataset_path"])
    elif config["model"] == "rnn" or config["model"] == "lstm":
        dataset = RNNDataset(path=config["dataset_path"])
    else: 
        raise ValueError(f"{config['model']} is not supported")

    #####################
    # initialize optuna #
    #####################

    study = optuna.create_study()
    study.optimize(func=Objective(config=config, dataset=dataset), n_trials=config["n_trials"])


if __name__ == "__main__":
    main()
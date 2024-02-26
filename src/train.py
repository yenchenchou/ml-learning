import argparse
import json

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.movielens import DataPrep, EnvInit, MovieLensDataset
from src.models import GMF, MLP, NCF

with open("src/params.json") as file:
    config = json.load(file)

env_init = EnvInit()
device = env_init.available_device()
seed = env_init.fix_seed(12345, device)


class Trainer:
    def __init__(self, model, criterion, optimizer, device, epoches) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoches = epoches
        self.device = device

    def fit(self, train_loader: DataLoader, eval_loader: DataLoader) -> torch.Tensor:
        train_num_batch = len(train_loader)
        eval_num_batch = len(eval_loader)
        self.model = self.model.to(device)
        for epoch in range(self.epoches):
            train_loss = 0
            self.model.train()
            for user_idxs, item_idxs, target in tqdm(train_loader):
                user_idxs = user_idxs.to(self.device)
                item_idxs = item_idxs.to(self.device)
                target = target.to(self.device)
                preds = self.model(user_idxs, item_idxs)
                loss = self.criterion(preds, target)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss += loss.item()
            train_loss /= train_num_batch

            self.model.eval()
            with torch.no_grad():
                eval_loss = 0
                for user_idxs, item_idxs, target in eval_loader:
                    user_idxs = user_idxs.to(self.device)
                    item_idxs = item_idxs.to(self.device)
                    target = target.to(self.device)
                    preds = self.model(user_idxs, item_idxs)
                    loss = self.criterion(preds, target)
                    eval_loss += loss.item()
                eval_loss /= eval_num_batch

            print(
                f"Epoch: {epoch+1} -- Train Loss: {train_loss} -- Eval Loss: {eval_loss}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Recommendation System Reimplementation on MovieLens"
    )
    parser.add_argument(
        "--neg_samp_ratio",
        type=int,
        default=4,
        help="Specify the negative sampling ratio, negative : positive = 4 : 1 (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "gmf", "ncf"],
        default="mlp",
        help="Specify which model sampling ratio, negative : positive = 4 : 1 (default: MLP)",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v0.1.0"],
        default="v0.1.0",
        help="Specify which model version",
    )
    args = parser.parse_args()
    prep = DataPrep()
    df_train, df_eval, df_test = prep()
    cfg = config[f"{args.model}-{args.version}"]
    train_data = MovieLensDataset(df=df_train, neg_sample_ratio=cfg["neg_samp_ratio"])
    eval_data = MovieLensDataset(df=df_eval, neg_sample_ratio=cfg["neg_samp_ratio"])
    # test_data = MovieLensDataset(df=df_test, neg_sample_ratio=cfg["neg_samp_ratio"])
    train_dataloader = DataLoader(train_data, batch_size=cfg["batch"], shuffle=True)
    eval_dataloader = DataLoader(eval_data, batch_size=cfg["batch"], shuffle=False)
    print(type(cfg["reg_layers"]))
    if args.model == "gmf":
        model = GMF(
            num_users=train_data.uniq_users.size,
            num_items=train_data.uniq_items.size,
            mf_dim=cfg["mf_dim"],
            reg_layers=cfg["reg_layers"],
        )
    elif args.model == "mlp":
        model = MLP(
            num_users=train_data.uniq_users.size,
            num_items=train_data.uniq_items.size,
            mlp_layers=cfg["mlp_layers"],
            reg_layers=cfg["reg_layers"],
        )
    elif args.model == "ncf":
        model = NCF(
            num_users=train_data.uniq_users.size,
            num_items=train_data.uniq_items.size,
            mf_dim=cfg["mf_dim"],
            mlp_layers=cfg["mlp_layers"],
            reg_layers=cfg["reg_layers"],
        )
    print(model)
    train = Trainer(
        model=model,
        criterion=nn.BCELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg["lr"]),
        device=device,
        epoches=cfg["epoches"],
    )
    train.fit(train_loader=train_dataloader, eval_loader=eval_dataloader)

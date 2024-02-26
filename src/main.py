import argparse
import json

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.movielens import DataPrep, EnvInit, MovieLensDataset
from src.models import GMF

with open("src/params.json") as file:
    config = json.load(file)

env_init = EnvInit()
device = env_init.available_device()
seed = env_init.fix_seed(12345)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Recommendation System Reimplementation on MovieLens"
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=["prep", "train", "test"],
        default="train",
        help="Specify the mode of the program (default: train)",
    )
    parser.add_argument(
        "--neg_sampling_ratio",
        type=int,
        default=4,
        help="Specify the negative sampling ratio, negative : positive = 4 : 1 (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["MLP", "NFM", "NCF"],
        default="MLP",
        help="Specify which model sampling ratio, negative : positive = 4 : 1 (default: MLP)",
    )
    args = parser.parse_args()
    if args.mode == "train":
        prep = DataPrep()
        df_train, df_eval, df_test = prep()
        train_data = MovieLensDataset(df=df_train, neg_sample_ratio=4)
        eval_data = MovieLensDataset(df=df_eval, neg_sample_ratio=4)
        test_data = MovieLensDataset(df=df_test, neg_sample_ratio=4)
        train_dataloader = DataLoader(
            train_data, batch_size=config["gmf-v0.1.0"]["batch"], shuffle=True
        )
        eval_dataloader = DataLoader(
            eval_data, batch_size=config["gmf-v0.1.0"]["batch"], shuffle=False
        )
        model = GMF(
            num_users=train_data.uniq_items.size,
            num_items=train_data.uniq_items.size,
            mf_dim=config["gmf-v0.1.0"]["embedding_dim"],
        )
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["gmf-v0.1.0"]["lr"])
        model = model.to(device)
        for epoch in range(config["gmf-v0.1.0"]["epoches"]):
            model.train()
            train_loss = 0
            train_num_batches = len(train_dataloader)
            train_pbar = tqdm(train_dataloader)
            for batch, (user_idxs, item_idxs, labels) in enumerate(train_pbar):
                optimizer.zero_grad()
                user_idxs = user_idxs.to(device)
                item_idxs = item_idxs.to(device)
                labels = labels.to(device)
                pred = model(user_idxs, item_idxs)
                loss = loss_fn(pred, labels)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss /= train_num_batches
            model.eval()
            eval_num_batches = len(eval_dataloader)
            eval_loss = 0
            with torch.no_grad():
                for batch, (user_idxs, item_idxs, labels) in enumerate(eval_dataloader):
                    user_idxs = user_idxs.to(device)
                    item_idxs = item_idxs.to(device)
                    labels = labels.to(device)
                    pred = model(user_idxs, item_idxs)
                    eval_loss += loss_fn(pred, labels).item()
            eval_loss /= eval_num_batches
            print(f"Train Avg loss: {train_loss}. Eval Avg loss: {eval_loss}")

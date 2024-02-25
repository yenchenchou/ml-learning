import argparse
import json

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.movielens import DataPrep, EnvInit, MovieLensDataset
from src.models import GMF, MLP

with open("src/params.json") as file:
    config = json.load(file)

env_init = EnvInit()
device = env_init.available_device()
seed = env_init.fix_seed(12345, device)


# loss_fn = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=config["gmf-v0.1.0"]["lr"])
# model = model.to(device)
# for epoch in range(config["gmf-v0.1.0"]["epoches"]):
#     model.train()
#     train_loss = 0
#     train_num_batches = len(train_dataloader)
#     train_pbar = tqdm(train_dataloader)
#     for batch, (user_idxs, item_idxs, labels) in enumerate(train_pbar):
#         optimizer.zero_grad()
#         user_idxs = user_idxs.to(device)
#         item_idxs = item_idxs.to(device)
#         labels = labels.to(device)
#         pred = model(user_idxs, item_idxs)
#         loss = loss_fn(pred, labels)
#         train_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     train_loss /= train_num_batches
#     model.eval()
#     eval_num_batches = len(eval_dataloader)
#     eval_loss = 0
#     with torch.no_grad():
#         for batch, (user_idxs, item_idxs, labels) in enumerate(eval_dataloader):
#             user_idxs = user_idxs.to(device)
#             item_idxs = item_idxs.to(device)
#             labels = labels.to(device)
#             pred = model(user_idxs, item_idxs)
#             eval_loss += loss_fn(pred, labels).item()
#     eval_loss /= eval_num_batches
#     print(f"Train Avg loss: {train_loss}. Eval Avg loss: {eval_loss}")


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
        "--neg_sampling_ratio",
        type=int,
        default=4,
        help="Specify the negative sampling ratio, negative : positive = 4 : 1 (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["MLP", "GMF", "NCF"],
        default="MLP",
        help="Specify which model sampling ratio, negative : positive = 4 : 1 (default: MLP)",
    )
    args = parser.parse_args()
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
    if args.model == "GMF":
        model = GMF(
            num_user_embeddings=train_data.uniq_items.size,
            num_item_embeddings=train_data.uniq_items.size,
            embedding_dim=config["gmf-v0.1.0"]["embedding_dim"],
        )
    elif args.model == "MLP":
        model = MLP(
            num_user_embeddings=train_data.uniq_items.size,
            num_item_embeddings=train_data.uniq_items.size,
            # embedding_dim=config["gmf-v0.1.0"]["embedding_dim"],
            layers=config["mlp-v0.1.0"]["layers"],
        )
        train = Trainer(
            model=model,
            criterion=nn.BCELoss(),
            optimizer=torch.optim.Adam(
                model.parameters(), lr=config["mlp-v0.1.0"]["lr"]
            ),
            device=device,
            epoches=config["mlp-v0.1.0"]["epoches"],
        )
        train.fit(train_loader=train_dataloader, eval_loader=eval_dataloader)

import argparse
import json

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.retrieval import (
    RetrievalHitRate,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)
from tqdm import tqdm

from src.data.movielens import DataPrep, EnvInit, MovieLensDataset
from src.models import GMF, MLP, NCF

with open("src/params.json") as file:
    config = json.load(file)

env_init = EnvInit()
device = env_init.available_device()
seed = env_init.fix_seed(12345, device)
writer = SummaryWriter(log_dir="data/runs/ncf_v0.1.0")


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device: str,
        epoches: int,
        top_k: int,
        model_path: str,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoches = epoches
        self.device = device
        self.hit = RetrievalHitRate(top_k=top_k)
        self.recall = RetrievalRecall(top_k=top_k)
        self.precision = RetrievalPrecision(top_k=top_k, adaptive_k=True)
        self.mrr = RetrievalMRR(top_k=top_k)
        self.ndcg = RetrievalNormalizedDCG(top_k=top_k)
        self.top_k = top_k
        self.model_path = model_path

    def fit(self, train_loader: DataLoader, eval_loader: DataLoader) -> torch.Tensor:
        best_ndcg = 0
        train_num_batch = len(train_loader)
        eval_num_batch = len(eval_loader)
        self.model = self.model.to(device)
        for epoch in range(self.epoches):
            train_loss = 0
            self.model.train()
            for name, param in self.model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
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
                preds_list = []
                for user_idxs, item_idxs, target in eval_loader:
                    user_idxs = user_idxs.to(self.device)
                    item_idxs = item_idxs.to(self.device)
                    target = target.to(self.device)
                    preds = self.model(user_idxs, item_idxs)
                    loss = self.criterion(preds, target)
                    eval_loss += loss.item()
                    preds_list.extend(preds)
                eval_loss /= eval_num_batch
                preds_list = torch.tensor(preds_list, dtype=torch.float32)
                hitrate = self.hit(
                    preds_list,
                    eval_loader.dataset.labels,
                    indexes=eval_loader.dataset.users.to(torch.long),
                ).item()
                precision = self.precision(
                    preds_list,
                    eval_loader.dataset.labels,
                    indexes=eval_loader.dataset.users.to(torch.long),
                ).item()
                recall = self.recall(
                    preds_list,
                    eval_loader.dataset.labels,
                    indexes=eval_loader.dataset.users.to(torch.long),
                ).item()
                mrr = self.mrr(
                    preds_list,
                    eval_loader.dataset.labels,
                    indexes=eval_loader.dataset.users.to(torch.long),
                ).item()
                ndcg = self.precision(
                    preds_list,
                    eval_loader.dataset.labels,
                    indexes=eval_loader.dataset.users.to(torch.long),
                ).item()
            writer.add_scalar("Train/Loss", train_loss, epoch)
            writer.add_scalar("Eval/Loss", eval_loss, epoch)
            writer.add_scalar(f"Eval/HitRate@{self.top_k}", round(hitrate, 6), epoch)
            writer.add_scalar(
                f"Eval/Precision@{self.top_k}", round(precision, 6), epoch
            )
            writer.add_scalar(f"Eval/Recall@{self.top_k}", round(recall, 6), epoch)
            writer.add_scalar(f"Eval/MRR@{self.top_k}", round(mrr, 6), epoch)
            writer.add_scalar(f"Eval/NDCG@{self.top_k}", round(ndcg, 6), epoch)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                torch.save(model.state_dict(), self.model_path)
                print(f"Save model to {self.model_path}")
            print(
                f"Epoch: {epoch+1} "
                + f"-- Train Loss: {train_loss:.6f} "
                + f"-- Eval Loss: {eval_loss:.6f} "
                + f"-- Eval HitRate@{self.top_k}: {hitrate:.6f} "
                + f"-- Eval Precision@{self.top_k}: {precision:.6f} "
                + f"-- Eval Recall@{self.top_k}: {recall:.6f} "
                + f"-- Eval MRR@{self.top_k}: {mrr:.6f} "
                + f"-- Eval NDCG@{self.top_k}: {ndcg:.6f}"
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
    user_idxs, item_idxs, target = next(iter(eval_dataloader))
    writer.add_graph(model, (user_idxs, item_idxs), True)
    train = Trainer(
        model=model,
        criterion=nn.BCELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=cfg["lr"]),
        device=device,
        epoches=cfg["epoches"],
        top_k=3,
        model_path=f"data/models/{args.model}-v0.1.0.pt",
    )
    train.fit(train_loader=train_dataloader, eval_loader=eval_dataloader)
    writer.close()

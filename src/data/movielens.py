import random
from ast import literal_eval
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class EnvInit:
    def available_device(self) -> Any:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = "cpu"
        return device

    def fix_seed(self, seed: int, device: str = None) -> int:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if torch == device:
            torch.mps.seed(seed)
        return seed


class DataFetcher:
    def load_train_eval(self) -> pl.DataFrame:
        return pl.read_csv(
            "data/2017-ncf-paper-data/ml-1m.train.rating",
            separator="\t",
            has_header=False,
            new_columns=["user_id", "movie_id", "rating", "timestamp"],
            schema={
                "user_id": pl.Int32,
                "movie_id": pl.Int32,
                "rating": pl.Int32,
                "timestamp": pl.Int64,
            },
            # n_rows=1000,
        )

    def load_test_pos(self):
        df = pl.read_csv(
            "data/2017-ncf-paper-data/ml-1m.test.rating",
            separator="\t",
            has_header=False,
            new_columns=["user_id", "movie_id", "rating", "timestamp"],
            schema={
                "user_id": pl.Int32,
                "movie_id": pl.Int32,
                "rating": pl.Int32,
                "timestamp": pl.Int64,
            },
        )
        df = df.drop("timestamp")
        return df.with_columns(pl.lit(1).alias("rating"))

    def load_test_neg(self):
        neg_users, neg_items = [], []
        with open("data/2017-ncf-paper-data/ml-1m.test.negative", "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user = literal_eval(arr[0])[0]
                negatives = [int(item) for item in arr[1:]]
                users = [user] * len(negatives)
                neg_users.extend(users)
                neg_items.extend(negatives)
                line = f.readline()
        return pl.DataFrame(
            {
                "user_id": neg_users,
                "movie_id": neg_items,
                "rating": [0] * len(neg_items),
            },
            schema={"user_id": pl.Int32, "movie_id": pl.Int32, "rating": pl.Int32},
        )

    def __call__(self) -> tuple:
        df_train_eval = self.load_train_eval()
        df_test_pos = self.load_test_pos()
        df_test_neg = self.load_test_neg()
        df_test = pl.concat([df_test_pos, df_test_neg], how="vertical")
        return (df_train_eval, df_test)


class DataCleaner:
    def sort_df(self, df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        return df.sort(["user_id", "timestamp"])

    def __call__(self, df: pl.DataFrame) -> Any:
        df = self.sort_df(df)
        return df


class DataSpliter:
    def split_df(
        self, df: pl.DataFrame, method: str, k: int
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if method == "leave-k-out":
            df_eval = df.group_by(["user_id"], maintain_order=True).tail(n=k)
            df_train = df.join(df_eval, on=df.columns, how="anti")
        return (df_train, df_eval)

    def __call__(
        self, df: pl.DataFrame | pd.DataFrame, method: str = "leave-k-out", k: int = 1
    ) -> Any:
        df_train, df_eval = self.split_df(df, method, k)
        return (df_train, df_eval)


class DataPrep:
    def __init__(self) -> None:
        self.downloader = DataFetcher()
        self.cleaner = DataCleaner()
        self.spliter = DataSpliter()

    def __call__(self, split_method: str = "leave-k-out", k: int = 1):
        df_train_eval, df_test = self.downloader()
        df_train_eval = self.cleaner(df_train_eval)
        df_train, df_eval = self.spliter(df_train_eval, method=split_method, k=k)
        print("Data Prep Done!!")
        return df_train, df_eval, df_test


class MovieLensDataset(Dataset):
    def __init__(self, df: pl.DataFrame, neg_sample_ratio: int):
        self.get_components(df)
        self.users, self.items, self.labels = self.neg_sampling(df, neg_sample_ratio)

    def neg_sampling(self, df: pl.DataFrame, neg_sample_ratio: int) -> pl.DataFrame:
        """_summary_

        Args:
            df (pl.DataFrame): _description_
            ratio (int, optional): _description_. Defaults to 4.

        Returns:
            pl.DataFrame: _description_
        """
        users, items, labels = [], [], []
        user_item_set = set(zip(df["user_id"], df["movie_id"]))
        for u, i in df[["user_id", "movie_id"]].iter_rows():
            users.append(u)
            items.append(i)
            labels.append(1)
            for _ in range(neg_sample_ratio):
                neg_i = np.random.choice(self.uniq_items)
                while (u, neg_i) in user_item_set:
                    neg_i = np.random.choice(self.uniq_items)
                users.append(u)
                items.append(neg_i)
                labels.append(0)
        return (
            torch.tensor(users, dtype=torch.int32),
            torch.tensor(items, dtype=torch.int32),
            torch.tensor(labels, dtype=torch.float32),
        )

    def get_components(self, df: pl.DataFrame | pd.DataFrame) -> None:
        """_summary_
        Notice:
            The + 1 for user_id and movie_id as their IDs starts with 1.
        Args:
            df (pl.DataFrame | pd.DataFrame): _description_
        """
        self._uniq_users = df["user_id"].unique().to_numpy()
        self._uniq_items = df["movie_id"].unique().to_numpy()

    @property
    def uniq_items(self) -> np.array:
        return self._uniq_items

    @property
    def uniq_users(self) -> np.array:
        return self._uniq_users

    def __len__(self):
        assert len(self.users) == len(self.items)
        return len(self.users)

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.users[idx], self.items[idx]
        else:
            return self.users[idx], self.items[idx], self.labels[idx]

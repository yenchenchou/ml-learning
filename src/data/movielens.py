import random
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
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

    def fix_seed(self, seed: int) -> int:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        return seed


class DataFetcher:
    def load_df(self) -> pl.DataFrame:
        return pl.from_pandas(
            pd.read_table(
                "data/ml-1m/ratings.dat",
                header=None,
                sep="::",
                names=["user_id", "movie_id", "rating", "timestamp"],
                dtype={
                    "user_id": np.int32,
                    "movie_id": np.int32,
                    "rating": np.int32,
                    "timestamp": np.int64,
                },
                nrows=100,
            )
        )

    def __call__(self) -> Any:
        df = self.load_df()
        return df


class DataCleaner:
    def sort_df(self, df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        return df.sort(["user_id", "timestamp"])

    def __call__(self, df: pl.DataFrame) -> Any:
        df = self.sort_df(df)
        return df


class DataSpliter:
    def split_df(
        self, df: pl.DataFrame, method: str, k: int
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        if method == "leave-k-out":
            df_eval_test = df.group_by(["user_id"], maintain_order=True).tail(n=2 * k)
            df_train = df.join(df_eval_test, on=df.columns, how="anti")
            df_eval = df_eval_test.group_by(["user_id"], maintain_order=True).head(n=k)
            df_test = df_eval_test.group_by(["user_id"], maintain_order=True).tail(n=k)
        return df_train, df_eval, df_test

    def __call__(
        self, df: pl.DataFrame | pd.DataFrame, method: str = "leave-k-out", k: int = 1
    ) -> Any:
        df_train, df_eval, df_test = self.split_df(df, method, k)
        return (df_train, df_eval, df_test)


class FeatEngineer:
    def __init__(self) -> None:
        pass

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

    def neg_sampling(self, df: pl.DataFrame, ratio: int) -> pl.DataFrame:
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
            for _ in range(ratio):
                neg_i = np.random.choice(self.uniq_items)
                while (u, neg_i) in user_item_set:
                    neg_i = np.random.choice(self.uniq_items)
                users.append(u)
                items.append(neg_i)
                labels.append(0)
        return (
            torch.tensor(users, dtype=torch.int32),
            torch.tensor(items, dtype=torch.int32),
            torch.tensor(labels, dtype=torch.int8),
        )


# class DataPrep:
#     def __init__(self) -> None:
#         self.downloader = DataFetcher()
#         self.cleaner = DataCleaner()
#         self.featengineer = FeatEngineer()
#         self.spliter = DataSpliter()

#     def __call__(self):
#         df = self.downloader()
#         df = self.cleaner(df)
#         df_train, df_eval, df_test = self.spliter(df, method="leave-k-out", k=1)
#         print("Done!!")


class MovieLensDataset(Dataset):
    def __init__(
        self, users: torch.tensor, items: torch.tensor, labels: torch.tensor = None
    ):
        self.users = users
        self.items = items
        if labels is not None:
            self.labels = None
        else:
            self.labels = labels

    def __len__(self):
        assert len(self.users) == len(self.items)
        return len(self.users)

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.users[idx], self.items[idx]
        else:
            return self.users[idx], self.items[idx], self.labels[idx]

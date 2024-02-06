import random
from typing import Any
import polars as pl
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
    def load_df(self, **kwargs) -> pl.DataFrame:
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
            df_train = df.join(df_eval_test, on=self.df.columns, how="anti")
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
        self._uniq_users = df["user_id"].unique().to_numpy()
        self._uniq_items = df["movie_id"].unique().to_numpy()

    @property
    def uniq_items(self):
        return self._uniq_items

    @property
    def uniq_users(self):
        return self._uniq_users

    def neg_sampling(self, df: pl.DataFrame, ratio: int = 4) -> pl.DataFrame:
        """_summary_

        Args:
            df (pl.DataFrame): _description_
            ratio (int, optional): _description_. Defaults to 4.

        Returns:
            pl.DataFrame: _description_
        """
        pass

    def __call__(self, df: pl.DataFrame) -> Any:
        self.get_components(df)
        return df


class DataPrep:
    def __init__(self) -> None:
        self.downloader = DataFetcher()
        self.cleaner = DataCleaner()
        self.featengineer = FeatEngineer()
        self.spliter = DataSpliter()

    def __call__(self):
        df = self.extracter()
        df = self.cleaner(df)
        df_train, df_eval, df_test = self.split_df(df)
        print("Done!!")


class MovieLensDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        if self.label is None:
            return self.data[idx]
        else:
            return self.data[idx], self.label[idx]

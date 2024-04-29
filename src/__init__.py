# from src.data.movielens_ncf import (
#     DataCleaner,
#     DataFetcher,
#     DataLoader,
#     DataPrep,
#     DataSpliter,
#     EnvInit,
#     MovieLensDataset,
# )

# __all__ = [
#     "DataCleaner",
#     "DataSpliter",
#     "DataFetcher",
#     "DataLoader",
#     "MovieLensDataset",
#     "EnvInit",
#     "DataPrep",
# ]

import random
from typing import Any

import numpy as np
import torch
from torch import nn


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

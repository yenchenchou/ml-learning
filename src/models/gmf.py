import torch
from torch import nn


class GMF(nn.Module):
    def __init__(
        self, num_user_embeddings: int, num_item_embeddings: int, embedding_dim: int
    ):
        super(GMF, self).__init__()
        self.user_embeddings = nn.Embedding(
            num_embeddings=num_user_embeddings, embedding_dim=embedding_dim, norm_type=0
        )
        self.item_embeddings = nn.Embedding(
            num_embeddings=num_item_embeddings, embedding_dim=embedding_dim, norm_type=0
        )
        self.user_embeddings.weight.data = torch.nn.init.normal_(
            self.user_embeddings.weight.data
        )
        self.item_embeddings.weight.data = torch.nn.init.normal_(
            self.item_embeddings.weight.data
        )
        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()


def forward(self, user_idxs: torch.Tensor, item_idxs: torch.Tensor) -> torch.Tensor:
    user_embeddings = self.user_embeddings(user_idxs)
    item_embeddings = self.item_embeddings(item_idxs)
    element_product = torch.mul(user_embeddings, item_embeddings)
    logits = self.linear(element_product)
    pred = self.sigmoid(logits)
    return pred.view(-1)

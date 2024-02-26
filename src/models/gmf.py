import torch
from torch import nn


class GMF(nn.Module):
    def __init__(
        self, num_users: int, num_items: int, mf_dim: int, reg_layers: list[int]
    ):
        super(GMF, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, mf_dim, norm_type=reg_layers[0])
        self.item_embeddings = nn.Embedding(num_items, mf_dim, norm_type=reg_layers[1])
        self.user_embeddings.weight.data = torch.nn.init.normal_(
            self.user_embeddings.weight.data
        )
        self.item_embeddings.weight.data = torch.nn.init.normal_(
            self.item_embeddings.weight.data
        )
        self.final_linear = nn.Linear(in_features=mf_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        user_idxs: torch.Tensor,
        item_idxs: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings = self.user_embeddings(user_idxs)
        item_embeddings = self.item_embeddings(item_idxs)
        x = torch.mul(user_embeddings, item_embeddings)
        logits = self.final_linear(x)
        preds = self.sigmoid(logits)
        return preds.view(-1)

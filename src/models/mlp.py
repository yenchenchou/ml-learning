import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self, num_user_embeddings: int, num_item_embeddings: int, layers: list[int]
    ) -> None:
        super(MLP, self).__init__()
        self.user_embeddings = nn.Embedding(num_user_embeddings, layers[0] // 2)
        self.item_embeddings = nn.Embedding(num_item_embeddings, layers[0] // 2)
        self.user_embeddings.weight.data = nn.init.normal_(
            self.user_embeddings.weight.data
        )
        self.item_embeddings.weight.data = nn.init.normal_(
            self.item_embeddings.weight.data
        )
        self.layers = layers
        self.multi_linears = nn.ModuleList(
            [
                nn.Linear(in_feat, out_feat)
                for in_feat, out_feat in zip(self.layers, self.layers[1:])
            ]
        )
        self.last_linear = nn.Linear(in_features=layers[-1], out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_idxs: torch.Tensor, item_idxs: torch.Tensor) -> torch.Tensor:
        user_embeddings = self.user_embeddings(user_idxs)
        item_embeddings = self.item_embeddings(item_idxs)
        x = torch.cat([user_embeddings, item_embeddings], dim=-1)
        for fc_layer in self.multi_linears:
            x = fc_layer(x)
            x = nn.ReLU()(x)
        logits = self.last_linear(x)
        preds = self.sigmoid(logits)
        return preds.view(-1)

import torch
from torch import nn


class NCF(nn.Module):
    def __init__(
        self,
        num_users: torch.Tensor,
        num_items: torch.Tensor,
        mf_dim: int,
        mlp_layers: list[int],
        reg_layers: list[int],
    ) -> None:
        super(NCF, self).__init__()
        self.gmf_user_embed = nn.Embedding(num_users, mf_dim, norm_type=reg_layers[0])
        self.gmf_item_embed = nn.Embedding(num_items, mf_dim, norm_type=reg_layers[1])
        self.mlp_user_embed = nn.Embedding(
            num_users,
            mlp_layers[0] // 2,
            norm_type=reg_layers[2],
        )
        self.mlp_item_embed = nn.Embedding(
            num_items,
            mlp_layers[0] // 2,
            norm_type=reg_layers[3],
        )
        self.mlp_linears = nn.ModuleList(
            [
                nn.Linear(in_feat, out_feat)
                for in_feat, out_feat in zip(mlp_layers, mlp_layers[1:])
            ]
        )
        self.final_layer = nn.Linear(mf_dim + mlp_layers[-1], 1)
        self.logistic = nn.Sigmoid()

    def forward(
        self,
        user_idxs: torch.Tensor,
        item_idxs: torch.Tensor,
    ) -> torch.Tensor:
        gmf_user_emb = self.gmf_user_embed(user_idxs)
        gmf_item_emb = self.gmf_item_embed(item_idxs)
        mlp_user_emb = self.mlp_user_embed(user_idxs)
        mlp_item_emb = self.mlp_item_embed(item_idxs)
        gmf_x = torch.mul(gmf_user_emb, gmf_item_emb)
        mlp_x = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        for fc_layer in self.mlp_linears:
            mlp_x = fc_layer(mlp_x)
            mlp_x = nn.ReLU()(mlp_x)
        x = torch.cat([gmf_x, mlp_x], dim=-1)
        logits = self.final_layer(x)
        preds = self.logistic(logits)
        return preds.view(-1)

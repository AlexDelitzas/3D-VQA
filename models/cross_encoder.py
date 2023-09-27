"""
Co-Attention Fusion Model
"""

# PyTorch
import torch
import torch.nn as nn

# Type support
from typing import Tuple


class Sem_Encoder(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 pdrop: float = 0.1):
        super(Sem_Encoder, self).__init__()
        self.co_attention_encoder = nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=hidden_size,
                                                                                           nhead=num_heads,
                                                                                           dim_feedforward=2048,
                                                                                           activation="gelu",
                                                                                           batch_first=True),
                                                          num_layers=num_layers)

    def forward(self,
                lang_rep: torch.Tensor,
                obj_rep: torch.Tensor,
                lang_mask: torch.Tensor,
                obj_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        NS = obj_rep.shape[1]
        NL = lang_rep.shape[1]

        # co-attention
        obj_mask = torch.zeros(obj_rep.shape[0:2], dtype=torch.bool).cuda()
        padding_mask_all = torch.cat([lang_mask, obj_mask], dim=1).bool()
        rep_all = torch.cat([lang_rep, obj_rep], dim=1)  # <B, NS+NL, C=256>

        # <B, NS+NL, C=256>
        co_attention_output = self.co_attention_encoder(rep_all, src_key_padding_mask=padding_mask_all)
        scene_hidden_states = co_attention_output[:, NL:, :]  # <B, NS, C=256>
        lang_hidden_states = co_attention_output[:, 0:NL, :]

        # post process
        lang_feature = lang_hidden_states[:, 0, :]

        return lang_feature, scene_hidden_states  # output

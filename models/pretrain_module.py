"""
Module contains the model that is used in the pretraining phase.
Called internally by scripts/pretrain.py

- General Skeleton taken from: https://github.com/ATR-DBI/ScanQA/blob/main/models/qa_module.py
"""

# OS
import os

# Pytorch
import torch
import torch.nn as nn

# Own Dependencies
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.model_utils import _print_state_dict_shapes, init_model_from_weights

# Type support and readability improvements
from typing import Dict, Optional, OrderedDict, Tuple
from numpy import ndarray
from einops import repeat


# Constants
LANG_SIZE = 512
DEVICE = "cuda"


class PretrainNet(nn.Module):
    def __init__(self,
                 num_class: int,
                 input_feature_dim: int,
                 num_heading_bin: int,
                 num_size_cluster: int,
                 mean_size_arr: ndarray,
                 num_proposal: int = 256,
                 vote_factor: int = 1,
                 sampling: str = "vote_fps",
                 seed_feat_dim: int = 512,
                 proposal_size: int = 128,
                 pointnet_width: int = 1,
                 pointnet_depth: int = 2,
                 vote_radius: float = 0.3,
                 vote_nsample: int = 16,
                 # scene encoder
                 hidden_size: int = 128,
                 enc_num_heads: int = 8,
                 enc_num_layers: int = 1,
                 clip_model: nn.Module = None,
                 use_reference: bool = False
                 ) -> None:
        super().__init__() 

        # Option
        self.use_reference = use_reference

        # Needs CUDA because of PointNet++ CUDA layers
        self.device = DEVICE

        # Object detection
        self.detection_backbone = Pointnet2Backbone(input_feature_dim=input_feature_dim,
                                                    width=pointnet_width,
                                                    depth=pointnet_depth,
                                                    seed_feat_dim=seed_feat_dim)

        # PointNet++ pretraining with DepthContrast
        pretrained_path = os.path.join('./checkpoints', 'checkpoint-ep150.pth.tar')
        assert os.path.isfile(pretrained_path), \
               f"{pretrained_path} is not a file. " \
               f"This file should contain the pretrained weights of PointNet++ Encoder." \
               f"Check that the filesystem is correct"

        # Initialize Model weights.
        self.detection_backbone = init_model_from_weights(self.detection_backbone, torch.load(pretrained_path))

        # Hough voting
        self.voting_net = VotingModule(vote_factor, seed_feat_dim)

        # Vote aggregation and object proposal
        self.proposal_net = ProposalModule(num_class,
                                           num_heading_bin,
                                           num_size_cluster,
                                           mean_size_arr,
                                           num_proposal,
                                           sampling,
                                           seed_feat_dim=seed_feat_dim,
                                           proposal_size=proposal_size,
                                           radius=vote_radius,
                                           nsample=vote_nsample)

        self.object_feat_linear = nn.Sequential(
            nn.Linear(proposal_size, hidden_size),
            nn.GELU()
        )

        # Scene Encoder
        self.scene_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=enc_num_heads),
                                                   num_layers=enc_num_layers)

        # Linear Layers to project cls token
        self.scene_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, LANG_SIZE)
        )

        # Estimate confidence
        self.object_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1)
        )

        # Used Clip Model
        self.clip_model = clip_model
        self.device = 'cuda'

        # Classification Token as in ViT
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, data_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        #######################################
        #                                     #
        #           LANGUAGE BRANCH           #
        #                                     #
        #######################################

        # --------- LANGUAGE ENCODING ---------
        texts = data_dict['lang_tokens']

        # Don't update CLIP weights
        with torch.no_grad():
            text_features = self.clip_model.encode_text(texts).float()

        # ---------- IMAGE ENCODING ----------

        image = data_dict["img"].to(self.device)

        # Don't update CLIP weights
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        data_dict = self.detection_backbone(data_dict)
        data_dict, features, xyz = self.hough_vote_(data_dict)

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal_net(xyz, features, data_dict)

        #######################################
        #                                     #
        #             QA BACKBONE             #
        #                                     #
        #######################################

        # unpack outputs from detection branch
        object_feat = data_dict['aggregated_vote_features']  # batch_size, num_proposal, proposal_size (128)

        # --------- Scene Encoder ---------
        # Get the scene representation
        object_feat = self.object_feat_linear(object_feat)  # batch_size, num_proposal, hidden_size

        # add a learnable token for the final representation (ViT style)
        batch_size = object_feat.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        object_feat = torch.cat((cls_tokens, object_feat), dim=1)

        scene_rep = self.scene_head(self.scene_encoder(object_feat)[:, 0])  # Only cls_token

        #######################################
        #                                     #
        #          PROPOSAL MATCHING          #
        #                                     #
        #######################################
        
        if self.use_reference:
            # mask out invalid proposals
            object_conf_feat = (object_feat *
                                data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2))  # object_feat
            data_dict["cluster_ref"] = self.object_cls(object_conf_feat).squeeze(-1)

        #######################################
        #                                     #
        #          TEXT-IMAGE SIMILARITY      #
        #                                     #
        #######################################

        data_dict["scene_rep"] = scene_rep
        data_dict["image_rep"] = image_features
        data_dict["text_rep"] = text_features

        return data_dict

    def hough_vote_(self,
                    data_dict: OrderedDict[str, torch.Tensor]
                    ) -> Tuple[OrderedDict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        data_dict : Data Dictionary containing all model outputs and inputs

        Returns
        -------
        - data_dict: Updated Data Dict
        - features: Featuire
        - xyz-data encoded
        """
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]  # batch_size, seed_feature_dim, num_seed, (16, 256, 1024)
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz

        data_dict["seed_features"] = features

        #  (batch_size, vote_feature_dim, num_seed * vote_factor, (16, 256, 1024)
        xyz, features = self.voting_net(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features
        return data_dict, features, xyz

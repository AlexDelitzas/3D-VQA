"""
Module contains the model that is used in the training phase.
Called internally by scripts/train.py

- Modified from: https://github.com/ATR-DBI/ScanQA/blob/main/models/qa_module.py
"""

# OS
import os

import numpy as np
# Pytorch
import torch
import torch.nn as nn

# Own Dependencies
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.lang_module import LangModule
from models.cross_encoder import Sem_Encoder
from models.model_utils import _print_state_dict_shapes, init_model_from_weights

# Type support and readability improvements
from typing import Dict, Optional, OrderedDict, Tuple
from einops import repeat


# General Constants
LANG_SIZE = 512
DEVICE = "cuda"

# Scene Encoder Constants
SCENE_ENC_EMB_DIM = 256
SCENE_ENC_N_HEADS = 8
SCENE_ENC_LAYERS = 1


class ScanQA(nn.Module):
    def __init__(self,
                 num_answers: int,
                 # proposal
                 num_object_class: int,
                 input_feature_dim: int,
                 num_heading_bin: int,
                 num_size_cluster: int,
                 mean_size_arr: np.ndarray,
                 num_proposal: int = 256,
                 vote_factor: int = 1,
                 sampling: str = "vote_fps",
                 seed_feat_dim: int = 256,
                 proposal_size: int = 128,
                 pointnet_width: int = 1,
                 pointnet_depth: int = 2,
                 vote_radius: float = 0.3,
                 vote_nsample: int = 16,
                 # qa
                 answer_pdrop: float = 0.3,
                 mcan_num_layers: int = 2,
                 mcan_num_heads: int = 8,
                 mcan_pdrop: float = 0.1,
                 mcan_flat_mlp_size: int = 512,
                 mcan_flat_glimpses: int = 1,
                 mcan_flat_out_size: int = 1024,
                 # lang
                 lang_use_bidir: bool = False,
                 lang_num_layers: int = 1,
                 lang_emb_size: int = 300,
                 lang_pdrop: float = 0.1,
                 bert_model_name: Optional[str] = None,
                 clip_model_name: Optional[str] = None,
                 freeze_bert: bool = False,
                 finetune_bert_last_layer: bool = False,
                 # common
                 hidden_size: int = 128,
                 # option
                 use_object_mask: bool = False,
                 use_lang_cls: bool = False,
                 use_reference: bool = False,
                 use_answer: bool = False,
                 ) -> None:
        super().__init__() 

        # Option
        self.use_object_mask = use_object_mask
        self.use_lang_cls = use_lang_cls
        self.use_reference = use_reference
        self.use_answer = use_answer

        # Language encoding 
        self.lang_net = LangModule(num_object_class,
                                   use_lang_classifier=False,
                                   use_bidir=lang_use_bidir,
                                   num_layers=lang_num_layers,
                                   emb_size=lang_emb_size,
                                   hidden_size=hidden_size,
                                   pdrop=lang_pdrop,
                                   bert_model_name=bert_model_name,
                                   clip_model_name=clip_model_name,
                                   freeze_bert=freeze_bert,
                                   finetune_bert_last_layer=finetune_bert_last_layer)

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
        self.proposal_net = ProposalModule(num_object_class,
                                           num_heading_bin,
                                           num_size_cluster,
                                           mean_size_arr,
                                           num_proposal,
                                           sampling,
                                           seed_feat_dim=seed_feat_dim,
                                           proposal_size=proposal_size,
                                           radius=vote_radius,
                                           nsample=vote_nsample)
        
        # Feature projection
        self.lang_feat_linear = nn.Sequential(
                nn.Linear(LANG_SIZE, hidden_size),
                nn.GELU()
        )

        self.object_feat_linear = nn.Sequential(
                nn.Linear(proposal_size, SCENE_ENC_EMB_DIM),
                nn.GELU()
        )

        # Scene Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, SCENE_ENC_EMB_DIM))
        self.scene_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=SCENE_ENC_EMB_DIM,
                                                                              nhead=SCENE_ENC_N_HEADS),
                                                   num_layers=SCENE_ENC_LAYERS)

        # Linear Projection of Attended Scene Embeddings
        self.scene_feat_linear = nn.Sequential(
                nn.Linear(SCENE_ENC_EMB_DIM, hidden_size),
                nn.GELU()
        )

        # Fusion backbone
        self.fusion_backbone = Sem_Encoder(hidden_size,
                                           num_heads=mcan_num_heads,
                                           num_layers=mcan_num_layers,
                                           pdrop=mcan_pdrop)

        # Estimate confidence
        self.object_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 1)
        )

        # Language classifier
        self.lang_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_object_class)
        )

        # QA head
        self.answer_cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(answer_pdrop),
                nn.Linear(hidden_size, num_answers)
        )

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #######################################
        #                                     #
        #           LANGUAGE BRANCH           #
        #                                     #
        #######################################

        # --------- LANGUAGE ENCODING ---------
        data_dict = self.lang_net(data_dict)        

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

        # unpack outputs from question encoding branch
        #       -> word embeddings after LSTM (batch_size, num_words(max_question_length), hidden_size * num_dir)
        lang_feat = data_dict["lang_out"]
        lang_mask = data_dict["lang_mask"]  # word attention (batch, num_words)
        
        # unpack outputs from detection branch
        object_feat = data_dict['aggregated_vote_features']  # batch_size, num_proposal, proposal_size (128)
        if self.use_object_mask:
            object_mask = ~data_dict["bbox_mask"].bool().detach()  # batch, num_proposals
        else:
            object_mask = None            
        if lang_mask.dim() == 2:
            lang_mask = lang_mask
        if object_mask.dim() == 2:
            object_mask = object_mask

        # --------- QA BACKBONE ---------
        # Pre-process Language & Image Feature
        lang_feat = self.lang_feat_linear(lang_feat)
        object_feat = self.object_feat_linear(object_feat)  # batch_size, num_proposal, SCENE_ENC_EMB_DIM

        # Scene Encoder
        batch_size = object_feat.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        object_feat = torch.cat((cls_tokens, object_feat), dim=1)
        object_feat = self.scene_encoder(object_feat)
        object_feat = self.scene_feat_linear(object_feat)  # batch_size, num_proposal, hidden_size

        # QA Backbone (Fusion network)  # lang_feat, object_feat
        lang_feat, object_feat = self.fusion_backbone(lang_feat,
                                                      object_feat,
                                                      lang_mask,
                                                      object_mask)
        # object_feat: batch_size, num_proposal, hidden_size
        # lang_feat: batch_size, num_words, hidden_size

        #######################################
        #                                     #
        #          PROPOSAL MATCHING          #
        #                                     #
        #######################################
        if self.use_reference:
            # mask out invalid proposals
            object_conf_feat = object_feat[:, 1:, :] * data_dict['objectness_scores'].max(2)[1].float().unsqueeze(2)
            data_dict["cluster_ref"] = self.object_cls(object_conf_feat).squeeze(-1)

        #######################################
        #                                     #
        #           LANGUAGE BRANCH           #
        #                                     #
        #######################################
        if self.use_lang_cls:
            data_dict["lang_scores"] = self.lang_cls(lang_feat)  # batch_size, num_object_classes

        #######################################
        #                                     #
        #          QUESTION ANSERING          #
        #                                     #
        #######################################
        if self.use_answer:
            data_dict["answer_scores"] = self.answer_cls(lang_feat)  # batch_size, num_answers

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

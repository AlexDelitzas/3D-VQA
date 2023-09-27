"""
Language Model for ScanQA

-> Copied and modified from: https://github.com/ATR-DBI/ScanQA/blob/main/models/lang_module.py
"""

# OS
import os
import sys

# PyTorch
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.getcwd(), "lib"))
from lib.qa_helper import *

# CLIP
from transformers import CLIPTextModel, CLIPTextConfig

# Type support and readability improvements
from typing import Dict, Optional, OrderedDict, Tuple


class LangModule(nn.Module):
    def __init__(self,
                 num_object_class: int,
                 use_lang_classifier: bool = True,
                 use_bidir: bool = False,
                 num_layers: int = 1,
                 emb_size: int = 300,
                 hidden_size: int = 256,
                 pdrop: float = 0.1,
                 word_pdrop: float = 0.1,
                 bert_model_name: Optional[str] = None,
                 clip_model_name: Optional[str] = None,
                 freeze_bert: bool = False,
                 finetune_bert_last_layer: bool = False
                 ) -> None:
        super().__init__() 

        # Init Attributes
        self.num_object_class = num_object_class
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.num_layers = num_layers
        self.bert_model_name = bert_model_name
        self.use_bert_model = bert_model_name is not None
        self.use_clip_model = clip_model_name is not None
        
        if self.use_bert_model:
            from transformers import AutoModel 
            self.bert_model = AutoModel.from_pretrained(bert_model_name)
            assert not (freeze_bert and finetune_bert_last_layer)
            if freeze_bert:
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            elif finetune_bert_last_layer:
                for param in self.bert_model.parameters():
                    param.requires_grad = False
                if hasattr(self.bert_model, 'encoder'):
                    for param in self.bert_model.encoder.layer[-1].parameters():
                        param.requires_grad = True
                else:  # distill-bert
                    for param in self.bert_model.transformer.layer[-1].parameters():
                        param.requires_grad = True
        elif self.use_clip_model:
            configuration = CLIPTextConfig()
            self.clip_model = CLIPTextModel(configuration).from_pretrained("openai/clip-vit-base-patch32")

        self.word_drop = nn.Dropout(pdrop)

        lang_size = hidden_size * 2 if use_bidir else hidden_size

        # Language classifier
        #   num_object_class -> 18
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Dropout(p=pdrop),
                nn.Linear(lang_size, num_object_class),
            )

    def make_mask(self, feature: torch.Tensor) -> torch.BoolTensor:
        """
        return a mask that is True for zero values and False for other values.
        """
        return feature.abs().sum(dim=-1) == 0

    def forward(self, data_dict: Dict[str, torch.Tensor]):
        """
        encode the input descriptions
        """

        if hasattr(self, 'bert_model'):
            print('not correct')
            word_embs = self.bert_model(**data_dict["lang_feat"])
            word_rep = word_embs.last_hidden_state  # batch_size, MAX_TEXT_LEN (32), bert_embed_size
        elif hasattr(self, 'clip_model'):
            with torch.no_grad():
                word_embs = self.clip_model(**data_dict["lang_feat"])
            word_rep = word_embs.last_hidden_state
        else:
            word_embs = data_dict["lang_feat"]  # batch_size, MAX_TEXT_LEN (32), glove_size
            raise NotImplementedError("word_rep is not implemented in this 'if' branch and this would cause an error "
                                      "in the next lines -> ABORT")

        # dropout word embeddings
        word_rep = self.word_drop(word_rep)
        lang_output = word_rep

        # encode description
        data_dict["lang_out"] = lang_output  # batch_size, num_words(max_question_length), hidden_size * num_dir


        # store the encoded language features
        data_dict["lang_embed"] = word_embs.pooler_output  # word_rep[:,-1,:]
        if self.use_bert_model or self.use_clip_model:
            # batch_size, num_words (max_question_length)
            data_dict["lang_mask"] = ~data_dict["lang_feat"]["attention_mask"][:, :lang_output.shape[1]].bool()
        else:
            data_dict["lang_mask"] = self.make_mask(lang_output) # batch_size, num_words (max_question_length)

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])
        return data_dict

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import clip

from typing import Tuple, Dict, Any, Tuple, OrderedDict, List

from torch.utils.data import DataLoader
from datetime import datetime

sys.path.append(os.path.join(os.getcwd()))
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.pretrain_dataset import PretrainDataset
from lib.pretrain_solver import PretrainSolver
from lib.config import CONF
from models.pretrain_module import PretrainNet
from models.model_utils import get_num_params


SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

IMG_PATH = './data/topdown_imgs/'
if not os.path.exists(IMG_PATH):
    IMG_PATH = CONF.PATH.TOPDOWN
    assert os.path.exists(IMG_PATH), f"{IMG_PATH} does not exist. Please specify the correct location of the 3D Scans"

# constants
DC = ScannetDatasetConfig()


def get_dataloader(args: argparse,
                   scanrefer,
                   all_scene_list,
                   topdown_imgs,
                   split: str,
                   clip_preprocess
                   ) -> Tuple[PretrainDataset, DataLoader]:
    # Get Dataset
    dataset = PretrainDataset(scanrefer=scanrefer[split],
                              scanrefer_all_scene=all_scene_list,
                              topdown_imgs=topdown_imgs[split],
                              split=split,
                              clip_preprocess=clip_preprocess,
                              num_points=args.num_points,
                              use_height=(not args.no_height))

    # Get Dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return dataset, dataloader


def get_model(args: argparse,
              clip_model: nn.Module
              ) -> PretrainNet:
    """
    Initiate Model
    Parameters
    ----------
    args : Argparsed Namespace containing the parsed values as attributes
    clip_model : CLIP Text and Image Encoder

    Returns
    -------
    model: Initiated Model
    """

    # initiate model
    input_channels = int(not args.no_height)
    model = PretrainNet(num_class=DC.num_class,
                        input_feature_dim=input_channels,
                        num_heading_bin=DC.num_heading_bin,
                        num_size_cluster=DC.num_size_cluster,
                        mean_size_arr=DC.mean_size_arr,
                        num_proposal=args.num_proposals,
                        hidden_size=args.hidden_size,
                        enc_num_heads=args.enc_num_heads,
                        enc_num_layers=args.enc_num_layers,
                        clip_model=clip_model)
    
    # to CUDA
    model = model.cuda()

    return model


def get_solver(args: argparse,
               dataloader: Dict[str, DataLoader],
               clip_model: nn.Module
               ) -> Tuple[PretrainSolver, int, str]:
    model = get_model(args, clip_model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag:
            stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    loss_weights = dict()

    # Original
    loss_weights['vote_loss'] = args.vote_loss_weight
    loss_weights['objectness_loss'] = args.objectness_loss_weight
    loss_weights['box_loss'] = args.box_loss_weight
    loss_weights['sem_cls_loss'] = args.sem_cls_loss_weight
    loss_weights['ref_loss'] = args.ref_loss_weight
    loss_weights['lang_loss'] = args.lang_loss_weight
    loss_weights['answer_loss'] = args.answer_loss_weight

    # Similarity
    image_sim_weight = args.image_similarity_weight
    text_sim_weight = args.text_similarity_weight
    loss_weights["image_loss"] = image_sim_weight / (image_sim_weight + text_sim_weight)
    loss_weights["text_loss"] = text_sim_weight / (image_sim_weight + text_sim_weight)
    loss_weights["clip_loss"] = args.clip_loss_weight
    loss_weights["object_detection_loss"] = args.object_detection_loss_weight

    solver = PretrainSolver(model=model,
                            config=DC,
                            dataloader=dataloader,
                            optimizer=optimizer,
                            stamp=stamp,
                            val_step=args.val_step,
                            detection=not args.no_detection,
                            reference=not args.no_reference,
                            use_lang_classifier=not args.no_lang_cls,
                            lr_decay_step=args.lr_decay_step,
                            lr_decay_rate=args.lr_decay_rate,
                            bn_decay_step=args.bn_decay_step,
                            bn_decay_rate=args.bn_decay_rate,
                            loss_weights=loss_weights)

    num_params = get_num_params(model)

    return solver, num_params, root


def save_info(args: argparse,
              root: str,
              num_params: int,
              train_dataset: Any,
              val_dataset: Any
              ) -> None:
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def get_scannet_scene_list(split: str):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META,
                                                                     "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer(scanrefer_train,
                  scanrefer_val,
                  num_scenes: int):
    # get initial scene list
    train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
    val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
    if num_scenes == -1: 
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes
    
    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list


def get_topdownimgs() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load Top-Down Images
    Returns
    -------

    """
    img_train = {}
    for image in os.listdir(os.path.join(IMG_PATH, 'train')):
        img_fn = os.path.join(IMG_PATH, 'train', image)
        img_train[image.split('.jpg')[0]] = img_fn 
    img_val = {}
    for image in os.listdir(os.path.join(IMG_PATH,'val')):
        img_fn = os.path.join(IMG_PATH, 'val', image)
        img_val[image.split('.jpg')[0]] = img_fn

    return img_train, img_val


def pretrain(args: argparse):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)
    scanrefer = {"train": scanrefer_train,
                 "val": scanrefer_val}

    # Get TopDown Images
    img_train, img_val = get_topdownimgs()
    topdown_imgs = {"train": img_train,
                    "val": img_val}

    # Init train images
    try:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda', jit=False)  # Must set jit=False for training
    except:
        # In case your machine doesn't have access to the internet: Use cached weights
        clip_model, clip_preprocess = clip.load(os.path.join(CONF.PATH.DATA, "..", "ViT-B-32.pt"),
                                                device='cuda',
                                                jit=False)
    clip_model.eval()

    # Freeze CLIP
    for p in clip_model.parameters():
        p.requires_grad = False

    # Dataloaders
    # (Pre)Training
    train_dataset, train_dataloader = get_dataloader(args,
                                                     scanrefer,
                                                     all_scene_list,
                                                     topdown_imgs,
                                                     "train",
                                                     clip_preprocess)

    # Validation Dataset
    val_dataset, val_dataloader = get_dataloader(args,
                                                 scanrefer,
                                                 all_scene_list,
                                                 topdown_imgs,
                                                 "val",
                                                 clip_preprocess)
    dataloader = {"train": train_dataloader,
                  "val": val_dataloader}

    print("initializing...")
    solver, num_params, root = get_solver(args, dataloader, clip_model)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_workers", type=int, help="Number of Processes", default=4)

    # General
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)

    # Modality/ Input Related
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed_feat_dim", type=int, default=512)
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_false", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_false", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str,
                        help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")

    # Overall Loss weights
    parser.add_argument("--vote_loss_weight", type=float, help="vote_net loss weight", default=1.0)  # 1.0
    parser.add_argument("--objectness_loss_weight", type=float, help="objectness loss weight", default=0.5)  # 0.5
    parser.add_argument("--box_loss_weight", type=float, help="box loss weight", default=1.0)  # 1.0
    parser.add_argument("--sem_cls_loss_weight", type=float, help="sem_cls loss weight", default=0.1)  # 0.1
    parser.add_argument("--ref_loss_weight", type=float, help="reference loss weight", default=0.1)  # 0.1
    parser.add_argument("--lang_loss_weight", type=float, help="language loss weight", default=0.1)  # 0.1
    parser.add_argument("--answer_loss_weight", type=float, help="answer loss weight", default=0.1)  # 0.1
    parser.add_argument("--clip_loss_weight", type=float, help="CLIP loss", default=0.1)  # 0.05 * 2 (for each loss)
    parser.add_argument("--object_detection_loss_weight", type=float,
                        help="Object_detection loss for pretrain", default=1.0)  # 0.05

    # CLIP Similarity Weights
    parser.add_argument("--image_similarity_weight", type=float,
                        help="Image Similarity Weight - Softmax", default=1.0)  # 1.0
    parser.add_argument("--text_similarity_weight", type=float,
                        help="Text Similarity Weight - Softmax", default=1.0)  # 1.0

    # Optimizer Related
    parser.add_argument('--lr_decay_step', nargs='+', type=int, default=[100, 200])  # 15
    parser.add_argument("--lr_decay_rate", type=float, help="decay rate of learning rate", default=0.2) # 01, 0.2
    parser.add_argument('--bn_decay_step', type=int, default=20)
    parser.add_argument("--bn_decay_rate", type=float, help="bn rate", default=0.5)
    parser.add_argument("--max_grad_norm", type=float, help="Maximum gradient norm ", default=1.0)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=2000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)  # 1e-3
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)

    # Scene Encoder Related
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden layer size[default: 256]")
    parser.add_argument("--enc_num_layers", type=int, default=1, help="Scene Encoder layers")
    parser.add_argument("--enc_num_heads", type=int, default=8, help="Scene Encoder heads")
    parser.add_argument("--enc_dropout", type=int, default=0.1, help="Scene dropout")

    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    pretrain(args)

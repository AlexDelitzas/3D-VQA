"""
File contains utility functions for qa-module and pretrain-module
"""

from typing import OrderedDict, Dict, Tuple, Union, List, Optional

import torch
import torch.nn as nn

import numpy as np

import os


def get_num_params(model: nn.Module):
    """
    Get Number of Parameteres
    Parameters
    ----------
    model :

    Returns
    -------

    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def _print_state_dict_shapes(state_dict: OrderedDict[str, torch.Tensor]) -> None:
    """
    Prints Shapes of State Dict parameters
    Parameters
    ----------
    state_dict : State Dict of Model containing the string keys and nn.Parameter params

    Returns
    -------

    """
    print("Model state_dict:")
    for param_tensor in state_dict.keys():
        print(f"{param_tensor}:\t{state_dict[param_tensor].size()}")


def init_model_from_weights(model: nn.Module,
                            state_dict: Union[OrderedDict[str, OrderedDict[str, torch.Tensor]],
                                              OrderedDict[str, torch.Tensor]],
                            skip_layers: Optional[str] = None,
                            print_init_layers: bool = True,
                            verbose: bool = True
                            ) -> nn.Module:
    """
    Initialize the model from any given params file. This is particularly useful
    during the fine-tuning process or when we want to evaluate a model on a range
    of tasks.
    skip_layers:     string : layer names with this key are not copied
    print_init_layers:   print whether layer was init or ignored
                    indicates whether the layer name was copied or not
    """
    # whether it's a model from somewhere else or a model from this codebase
    try:
        state_dict = state_dict["model_state_dict"]  # model
    except:
        try:
            state_dict = state_dict["model"]  # model
        except:
            pass
            # State Dict is already StateDict
        pass

    all_layers = model.state_dict()
    init_layers = {layername: False for layername in all_layers}
    new_state_dict = {}
    for param_name in state_dict:
        if "module.trunk.0" not in param_name:
            continue
        param_data = param_name.split(".")
        newname = ""  # "backbone_net"
        for i in range(len(param_data[3:])):  # 3
            newname += ". " + param_data[i+3]
        newname = newname[1:]

        new_state_dict[newname] = state_dict[param_name]
    state_dict = new_state_dict

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    not_found, not_init = [], []
    for layername in all_layers.keys():
        if (
                skip_layers and len(skip_layers) > 0 and layername.find(skip_layers) >= 0
        ) or layername.find("num_batches_tracked") >= 0:
            if print_init_layers and (local_rank == 0):
                not_init.append(layername)
                print(f"Ignored layer:\t{layername}")
            continue
        if layername in state_dict:
            param = state_dict[layername]
            if not isinstance(param, torch.Tensor):
                param = torch.from_numpy(param)
            all_layers[layername].copy_(param)

            init_layers[layername] = True
            if print_init_layers and (local_rank == 0):
                print(f"Init layer:\t{layername}")
        else:
            not_found.append(layername)
            if print_init_layers and (local_rank == 0):
                print(f"Not found:\t{layername}")

    """####################### DEBUG ############################"""
    if verbose:
        _print_state_dict_shapes(model.state_dict())

    return model


def init_votenet_from_weights(model: nn.Module,
                              state_dict: Union[OrderedDict[str, torch.Tensor],
                                                Dict[str, OrderedDict[str, torch.Tensor]]],
                              skip_layers: Optional[bool] = None,
                              print_init_layers=True,
                              ) -> nn.Module:
    """
    Initialize the model from any given params file. This is particularly useful
    during the finetuning process or when we want to evaluate a model on a range
    of tasks.
    skip_layers:     string : layer names with this key are not copied
    print_init_layers:   print whether layer was init or ignored
                    indicates whether the layername was copied or not
    """
    # whether it's a model from somewhere else or a model from this codebase
    state_dict = state_dict["model_state_dict"]  # model

    all_layers = model.state_dict()
    init_layers = {layername: False for layername in all_layers}

    new_state_dict = {}
    for param_name in state_dict:
        param_data = param_name.split(".")
        if param_data[0] == 'vgen':
            param_data[0] = 'voting_net'
        elif param_data[0] == 'pnet':
            param_data[0] = 'proposal_net'
        elif param_data[0] == 'backbone_net':
            param_data[0] = 'detection_backbone'
        newname = param_data[0]  # "backbone_net"
        for i in range(len(param_data[1:])):
            newname += "." + param_data[i + 1]
        new_state_dict[newname] = state_dict[param_name]
    state_dict = new_state_dict

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    not_found, not_init = [], []
    for layername in all_layers.keys():
        if (
                skip_layers and len(skip_layers) > 0 and layername.find(skip_layers) >= 0
        ) or layername.find("num_batches_tracked") >= 0:
            if print_init_layers and (local_rank == 0):
                not_init.append(layername)
                print(f"Ignored layer:\t{layername}")
            continue
        if layername in state_dict:
            param = state_dict[layername]

            if not isinstance(param, torch.Tensor):
                param = torch.from_numpy(param)
            all_layers[layername].copy_(param)
            init_layers[layername] = True
            if print_init_layers and (local_rank == 0):
                print(f"Init layer:\t{layername}")
        else:
            not_found.append(layername)
            if print_init_layers and (local_rank == 0):
                print(f"Not found:\t{layername}")
    ####################### DEBUG ############################
    # _print_state_dict_shapes(model.state_dict())

    return model

import os
import shutil
import obonet
import torch
from torch.utils.data._utils.collate import default_collate
from num2words import num2words
from typing import Optional, Tuple
import pickle
from data_processing.dataset import CustomDataset, CustomDataCollator
import yaml
from torch.utils.data import DataLoader


def custom_collate_fn(batch, matching_modality='ss'):
    default_batch = default_collate(batch)
    if matching_modality == 'ss':
        return {'sequence_modality': default_batch[0],  'structure_modality': default_batch[1]}
    elif matching_modality == 'st':
        return {'sequence_modality': default_batch[0],  'text_modality': default_batch[1]}
    


def collate_fn_wrapper(collate_fn, *args):
    def wrapper(batch):
        return collate_fn(batch, *args)
    return wrapper


def get_collate_fn(matching_modality='ss'):
    wrapped_collate_fn = collate_fn_wrapper(custom_collate_fn, matching_modality)
    return wrapped_collate_fn


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{total_params:,}"


def print_all_modules(model):
    for name, module in model.named_modules():
        print(name)


def print_trainable_modules(model):
    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters()):
            print(f"Trainable module: {name}")



def save_ckp(state, checkpoint_dir, filename):
    """
    state: checkpoint we want to save
    checkpoint_path: path to save checkpoint
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = f"{filename}"
    torch.save(state, checkpoint_path)


def load_ckp(
    model: torch.nn.Module,
    filename: str,
    optimizer: Optional[torch.optim.Optimizer] = None, 
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
    model_only: bool = False, 
    device: str = "cpu",
    strict=True
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler], Optional[int], Optional[float]]:
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    
    checkpoint_fpath = f"{filename}"


    # Load checkpoint
    if not os.path.exists(checkpoint_fpath):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_fpath}")
    
    checkpoint = torch.load(checkpoint_fpath,  map_location=device, weights_only=True)

    model.load_state_dict(checkpoint['state_dict'], strict=strict)


    # initialize optimizer from checkpoint to optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        

    # initialize lr scheduler from checkpoint to optimizer
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # initialize valid_loss_min from checkpoint to valid_loss_min
    # valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss

    if model_only:
        return model
    
    return model, optimizer, lr_scheduler, checkpoint['epoch']#, valid_loss_min


def save_pickle(obj, filepath):
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object successfully saved to {filepath}.")
    except Exception as e:
        print(f"Error saving object: {e}")

def load_pickle(filepath):
    try:
        with open(filepath, 'rb') as file:
            obj = pickle.load(file)
        print(f"Object successfully loaded from {filepath}.")
        return obj
    except Exception as e:
        print(f"Error loading object: {e}")
        return None



def fuse_output(outputs_list, strategy="mean"):

    if strategy == "mean":
        fused_output = torch.mean(torch.stack(outputs_list, dim=0), dim=0)

    elif strategy == "sum":
        fused_output = torch.sum(torch.stack(outputs_list, dim=0), dim=0)

    elif strategy == "max":
        fused_output = torch.max(torch.stack(outputs_list, dim=0), dim=0).values

    elif strategy == "concat":
        fused_output = torch.cat(outputs_list, dim=-1)

    elif strategy == "weighted":
        stacked_outputs = torch.stack(outputs_list, dim=0)
        weights = torch.tensor([0.1, 0.3, 0.2, 0.4])
        fused_output = torch.tensordot(weights, stacked_outputs, dims=([0], [0]))
        
    return fused_output
        

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config



def load_graph(graph_pth="/home/fbqc9/Workspace/MCLLM_DATA/obo/go-basic.obo"):
    go_graph = obonet.read_obo(open(graph_pth, 'r'))

    accepted_edges = set()
    unaccepted_edges = set()

    for edge in go_graph.edges:
        if edge[2] == 'is_a' or edge[2] == 'part_of':
            accepted_edges.add(edge)
        else:
            unaccepted_edges.add(edge)
    go_graph.remove_edges_from(unaccepted_edges)
    return go_graph



def load_data(modality_pair, config, batch_size, device, shuffle, validation=False, meta=False):
    dataset = CustomDataset(modality_pair=modality_pair, config=config, validation=validation, meta=meta)
    collate_fxn = CustomDataCollator(modality_pair=modality_pair, device=device, meta=meta)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fxn)


def get_model_size_in_gb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()  # Total parameter size in bytes
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()  # Total buffer size in bytes

    total_size = param_size + buffer_size  # Total size in bytes
    return total_size / (1024 ** 3) 


def get_model_size_gb(model):
    total_params = sum(p.numel() for p in model.parameters())

    total_bytes = total_params * 4
    total_gb = total_bytes / (1024**3)  

    return total_gb
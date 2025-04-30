import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from Loss import  InfoNCELoss, DiversityLoss
from Metrics import PreRecF
from data_processing.dataset import CustomDataset, CustomDataCollator
from models.model import SeqBindClassifier
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils import get_model_size_gb, load_pickle, save_ckp, load_ckp
import yaml
import random
import numpy as np
from num2words import num2words
from Const import BASE_DATA_DIR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_data(modality_pair, config, batch_size, device, shuffle, validation=False):
    dataset = CustomDataset(modality_pair=modality_pair, config=config, validation=validation)
    print(modality_pair, len(dataset))
    collate_fxn = CustomDataCollator(modality_pair=modality_pair, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fxn)



def run_epoch(model, dataloader, labels, criteria, metrics, optimizer=None, is_training=False):
    
    keys = dataloader.keys()
    x_metrics = ["Precision", "Recall", "Fscore", "Smin", "Weighted Precision", "Weighted Recall", "Weighted Fscore", "Weighted Smin"]
    scores = {metric: {} for metric in x_metrics}
    
    losses = {f"{key}": 0 for key in keys}
    counts = {i: 0 for i in losses.keys()}
    
    for key in keys:
        mod1, mod2 = key.split("_")
        for metric in scores.keys():
            scores[metric][f"{mod1}"] = 0
            scores[metric][f"{mod2}"] = 0
            counts[f"{mod1}"] = 0
            counts[f"{mod2}"] = 0
    
    max_batch = max(len(data_loader) for data_loader in dataloader.values())
    dataloaders_iter = {modality_pair: iter(dataloader) for modality_pair, dataloader in dataloader.items()}
    
    with torch.set_grad_enabled(is_training):
        for _ in range(max_batch):
            modality_pairs = list(dataloaders_iter.keys())
            random.shuffle(modality_pairs)
            for modality_pair in modality_pairs:
                try:
                    pair1, pair2 = modality_pair.split("_")
                    batch = next(dataloaders_iter[modality_pair])
                    proteins = batch['protein']
                    
                    if is_training:
                        optimizer.zero_grad()
                    
                    encoder_outputs, classification_outputs, expert_outputs = model(batch)
                    
                    con_loss = criteria['contrastive'](encoder_outputs[f'{pair1}_modality'], encoder_outputs[f'{pair2}_modality'])
                    
                    labeled_indices = [i for i, protein in enumerate(proteins) if protein in labels]
                    if labeled_indices:
                        _labels = [labels[proteins[i]] for i in labeled_indices]
                        cls_labels = torch.tensor(_labels).float().to(args.device)
                        
                        cls_output_mod1 = classification_outputs[f'{pair1}_modality'][labeled_indices]
                        cls_output_mod2 = classification_outputs[f'{pair2}_modality'][labeled_indices]
                        
                        cls_loss = criteria["classification"](cls_output_mod1, cls_labels) + \
                                criteria["classification"](cls_output_mod2, cls_labels)
                                                
                        div_loss = criteria["diversity"](expert_outputs[f'{pair1}_modality']) + \
                                criteria["diversity"](expert_outputs[f'{pair2}_modality'])
                        
                        loss = cls_loss #+ 0.01 * div_loss #+  0.5 * con_loss
                        
                        counts[f'{pair1}'] += 1
                        counts[f'{pair2}'] += 1
                        
                        res1 = metrics["PreRecF"].compute_scores(cls_labels, cls_output_mod1)
                        res2 = metrics["PreRecF"].compute_scores(cls_labels, cls_output_mod2)
                        for metric in scores.keys():
                            scores[metric][f"{pair1}"] += res1[metric]
                            scores[metric][f"{pair2}"] += res2[metric]
                    
                    losses[modality_pair] += loss.item()
                    counts[modality_pair] += 1
                    
                    if is_training:
                        loss.backward()
                        optimizer.step()
                    
                except StopIteration:
                    continue
    
    losses = {modality_pair: value / counts[modality_pair] for modality_pair, value in losses.items()}
    scores = {metric: {modality: score / counts[modality] for modality, score in modality_scores.items()} 
              for metric, modality_scores in scores.items()}
    
    return {"Losses": losses, "Scores": scores}



def train(model, train_dataloader, val_dataloader, labels, criteria, metrics, optimizer, scheduler, current_epoch):

    if args.wandb:
        wandb.init(
            project="FunBind",
            name=f"{args.go_ontology}",
            config={
            "learning rate": args.lr,
            "epochs": args.epochs,
            "batch size": args.train_batch,
            "weight decay": args.weight_decay,
            "criteria": criteria,
            }
        )


    # Add a description
    wandb.notes = "Learning Rate:{}, Batch Size:{} Weight Decay:{}".format(args.lr, args.train_batch, args.weight_decay)
    

    for epoch in range(current_epoch, args.epochs):

        print("Training model, epoch {}".format(epoch))

        model.train()
        torch.autograd.set_detect_anomaly(True)

        train_results = run_epoch(model, train_dataloader, labels, criteria, metrics, optimizer, is_training=True)
        print(train_results)

        # validation
        model.eval()
        print("#######################")
        valid_results = run_epoch(model, val_dataloader, labels, criteria, metrics, is_training=False)
        print(valid_results)


        scheduler.step()

        results = {"Training loss": train_results["Losses"], "Training Score": train_results["Scores"], 
                   "Validation loss": valid_results["Losses"], "Validation Score": valid_results["Scores"]}


        if args.wandb:
            wandb.log({**results, "Epoch": epoch})

    
        if args.save_model:

            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict()
            }
            save_ckp(state=checkpoint, checkpoint_dir=ckp_dir, filename=ckp_file)



    if args.wandb:
        wandb.finish()



if __name__ == '__main__':

    parser = argparse.ArgumentParser('Pre-training', add_help=False)
    parser.add_argument('--no-cuda', action='store_true', default=False, help="Disables CUDA training.")
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.00002, help='Initial learning rate.')
    parser.add_argument('--train_batch', type=int, default=50, help='Training batch size.')
    parser.add_argument('--valid_batch', type=int, default=50, help='Validation batch size.')
    parser.add_argument('--wandb', default=False, help='Send to wandb')
    parser.add_argument('--save_model', default=False, help='Save model.')
    parser.add_argument('--load_weights', default=False, help='Load saved model.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer.')
    parser.add_argument('--go_ontology', type=str, choices=['CC', 'MF', 'BP'], default='CC', help="Specific Ontology")
    args = parser.parse_args()

    # parser.add_argument('--lr', type=float, default=0.00002, help='Initial learning rate.')


    set_seed(42) 

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.device = "cuda:0"
    else:
        args.device = "cpu"

    config = load_config('config.yaml')['config1']

    # batch_sizes = [round(batch_size_smallest * (size / dataset_sizes[0])) for size in dataset_sizes]
    train_dataloaders = {
        "Sequence_Text": load_data(modality_pair="Sequence_Text", config=config, batch_size=500, device=args.device, shuffle=True),
        "Sequence_Interpro": load_data(modality_pair="Sequence_Interpro", config=config, batch_size=580, device=args.device, shuffle=True),
        "Sequence_Structure": load_data(modality_pair="Sequence_Structure", config=config, batch_size=570, device=args.device, shuffle=True),
    }


    val_dataloaders = {
        "Sequence_Structure": load_data(modality_pair="Sequence_Structure", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
        "Sequence_Text": load_data(modality_pair="Sequence_Text", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
        "Sequence_Interpro": load_data(modality_pair="Sequence_Interpro", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
    }

    groundtruth = load_pickle(BASE_DATA_DIR + "/data/labels/{}_labels".format(args.go_ontology))
    info_acc = load_pickle(BASE_DATA_DIR + "/data/labels/{}_ia".format(args.go_ontology))
    

    model = SeqBindClassifier(config=config, go_ontology=args.go_ontology).to(args.device)
    _ckp_file = BASE_DATA_DIR + '/saved_models/pretrained_ontology.pt'
    model = load_ckp(filename=_ckp_file, model=model, model_only=True, strict=False) 
    print(model)


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters entire model: {num2words(total_params), total_params}")
    print("********************************************")

    sub_model = model.get_submodule("modality_encoder")
    total_params = sum(p.numel() for p in sub_model.parameters())
    print(f"Total parameters encoder: {num2words(total_params), total_params}") 
    print("********************************************")

    sub_model = model.get_submodule("modality_encoder").get_submodule("Sequence_modality")
    total_params = sum(p.numel() for p in sub_model.parameters())
    print(f"Total parameters sequence: {num2words(total_params), total_params}")
    print("********************************************")

    sub_model = model.get_submodule("modality_encoder").get_submodule("Structure_modality")
    total_params = sum(p.numel() for p in sub_model.parameters())
    print(f"Total parameters structure: {num2words(total_params), total_params}")
    print("********************************************")

    sub_model = model.get_submodule("modality_encoder").get_submodule("Text_modality")
    total_params = sum(p.numel() for p in sub_model.parameters())
    print(f"Total parameters text: {num2words(total_params), total_params}")
    print("********************************************")

    sub_model = model.get_submodule("modality_encoder").get_submodule("Interpro_modality")
    total_params = sum(p.numel() for p in sub_model.parameters())
    print(f"Total parameters interpro: {num2words(total_params), total_params}")
    print("********************************************")


    criteria = { 
            "contrastive": InfoNCELoss(logit_scale=1/0.07, device=args.device),
            "classification": torch.nn.BCELoss(), "diversity": DiversityLoss() }

    metrics = {
        "PreRecF": PreRecF(infor_accr=info_acc)
    }

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    ckp_dir = BASE_DATA_DIR + '/saved_models/'
    ckp_file = ckp_dir + "{}.pt".format(args.go_ontology)

    if args.load_weights and os.path.exists(ckp_file):
        print("Loading model checkpoint @ {}".format(ckp_file))
        model, optimizer, lr_scheduler, current_epoch = load_ckp(filename=ckp_file, model=model, 
                                                                 optimizer=optimizer, 
                                                                 lr_scheduler=lr_scheduler)
    else:
        current_epoch = 0
        min_val_loss = np.inf

    train(model=model, 
          train_dataloader=train_dataloaders, 
          val_dataloader=val_dataloaders, 
          labels=groundtruth,
          criteria=criteria, metrics= metrics, optimizer=optimizer, 
          scheduler=lr_scheduler, current_epoch=current_epoch)
    

    
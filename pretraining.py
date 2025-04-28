import argparse, os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from Loss import DiversityLoss, InfoNCELoss
from Metrics import Similarity
from data_processing.dataset import CustomDataset, CustomDataCollator
from data_processing.utils import load_pickle
from models.model import SeqBindPretrain
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, SequentialLR, LinearLR, CosineAnnealingLR
from utils import save_ckp, load_ckp
import yaml
import random
import numpy as np




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
    collate_fxn = CustomDataCollator(modality_pair=modality_pair, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fxn)


def train_epoch(model, train_dataloader, criteria, metrics, optimizer):

    model.train()

    keys = train_dataloader.keys()
        
    scores = {"Similarity": {i: 0 for i in keys}}
    losses = {f"{i}": 0 for i in keys}
    countings = {i: 0 for i in keys}

    
    max_batches = max(len(data_loader) for data_loader in train_dataloader.values())

    # Initialize iterators for each dataloader
    dataloaders_iter = {modality_pair: iter(dataloader) for modality_pair, dataloader in train_dataloader.items()}

    for mini_epoch in range(max_batches):
        for modality_pair, dataloader_iter in dataloaders_iter.items():
            # print(modality_pair)
            try:
                pair1, pair2 = modality_pair.split("_")
                batch = next(dataloader_iter)
                
                optimizer.zero_grad()

                outputs, expert_outputs = model(batch)

                con_loss = criteria[modality_pair](outputs[f'{pair1}_modality'], outputs[f'{pair2}_modality'])
                
                div_loss = criteria["diversity"](expert_outputs[f'{pair1}_modality']) + criteria["diversity"](expert_outputs[f'{pair2}_modality'])

                loss = con_loss + 0.1 * div_loss

                losses[modality_pair] += loss.item()
                countings[modality_pair] += 1

                loss.backward()
                optimizer.step()

                scores['Similarity'][modality_pair] += metrics["Similarity"][modality_pair](outputs[f'{pair1}_modality'], outputs[f'{pair2}_modality'])

            except StopIteration:
                continue
            except ValueError:
                continue


    losses = {_key: _value/countings[_key]  for _key, _value in losses.items()}
    scores = {metric: {modality_pair: (score / countings[modality_pair]) for modality_pair, score in modality_scores.items()} for metric, modality_scores in scores.items()}
    return {"Losses": losses, "Scores": scores}


def validate_epoch(model, valid_dataloader, criteria, metrics):
    model.eval()
    with torch.no_grad():

        keys = valid_dataloader.keys()
        
        scores = {"Similarity": {i: 0 for i in keys}}
        losses = {f"{i}": 0 for i in keys}
        countings = {i: 0 for i in keys}

        max_batches = max(len(data_loader) for data_loader in valid_dataloader.values())
        dataloaders_iter = {modality_pair: iter(dataloader) for modality_pair, dataloader in valid_dataloader.items()}

        for mini_epoch in range(max_batches):
            for modality_pair, dataloader_iter in dataloaders_iter.items():
                try:
                    pair1, pair2 = modality_pair.split("_")
                    batch = next(dataloader_iter)

                    outputs, expert_outputs = model(batch)

                    if modality_pair in criteria:
                        con_loss = criteria[modality_pair](outputs[f'{pair1}_modality'], outputs[f'{pair2}_modality'])
                    else:
                        con_loss = criteria['emergent_binding'](outputs[f'{pair1}_modality'], outputs[f'{pair2}_modality'])

                    div_loss = criteria["diversity"](expert_outputs[f'{pair1}_modality']) + criteria["diversity"](expert_outputs[f'{pair2}_modality'])

                    loss = con_loss + 0.1 * div_loss
                    losses[modality_pair] += loss.item()
                    countings[modality_pair] += 1

                    if modality_pair in metrics['Similarity']:
                        scores['Similarity'][modality_pair] += metrics["Similarity"][modality_pair](outputs[f'{pair1}_modality'], outputs[f'{pair2}_modality'])
                    else:
                        scores['Similarity'][modality_pair] += metrics["Similarity"]['emergent_binding'](outputs[f'{pair1}_modality'], outputs[f'{pair2}_modality'])

                except StopIteration:
                    continue
                

    scores = {metric: {modality_pair: score / countings[modality_pair] for modality_pair, score in modality_scores.items()} for metric, modality_scores in scores.items()}
    losses = {_key: _value/countings[_key]  for _key, _value in losses.items()}
    return {"Losses": losses, "Scores": scores}


def train(model, train_dataloader, val_dataloader, criteria, metrics, optimizer, scheduler, current_epoch):

    if args.wandb:
        wandb.init(
            project="FunBind",
            name="Pretraining",
            notes=f"Learning Rate: {args.lr}, "
                  f"Batch Size: {args.train_batch}, Weight Decay: {args.weight_decay}",
            config={
            "learning rate": args.lr,
            "epochs": args.epochs,
            "batch size": args.train_batch,
            "weight decay": args.weight_decay, 
            "criteria": criteria,
            }
        )
    

    for epoch in range(current_epoch, args.epochs):

        print("Training model, epoch {}".format(epoch))

        train_results = train_epoch(model, train_dataloader, criteria, metrics, optimizer)
        print(train_results)

        print("#######################")
        valid_results = validate_epoch(model, val_dataloader, criteria, metrics)
        print(valid_results)

        print("#####################################################################\n")
        

        scheduler.step()

        results = {"Training loss": train_results["Losses"], 
                    "Training Score": train_results["Scores"], 
                    "Validation loss": valid_results["Losses"], 
                    "Validation Score": valid_results["Scores"]}
    


        if args.wandb:
            wandb.log({**results, "Epoch": epoch})

    
        if args.save_model:

            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            save_ckp(state=checkpoint, checkpoint_dir=ckp_dir, filename=ckp_file)


    if args.wandb:
        wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Pre-training', add_help=False)
    parser.add_argument('--no-cuda', action='store_true', default=False, help="Disables CUDA training.")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
    parser.add_argument('--train_batch', type=int, default=512, help='Training batch size.')
    parser.add_argument('--valid_batch', type=int, default=512, help='Validation batch size.')
    parser.add_argument('--wandb', default=True, help='Send to wandb')
    parser.add_argument('--save_model', default=True, help='Save model.')
    parser.add_argument('--load_weights', default=True, help='Load saved model.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer.')

    args = parser.parse_args()

    set_seed(42) 

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.device = "cuda:1"
    else:
        args.device = "cpu"

    args.device = "cuda:1"

    config = load_config('config.yaml')['config_prostt5']


    train_dataloaders = {
        "Sequence_Structure": load_data(modality_pair="Sequence_Structure", config=config, batch_size=args.train_batch, device=args.device, shuffle=True),
        "Sequence_Text": load_data(modality_pair="Sequence_Text", config=config, batch_size=args.train_batch, device=args.device, shuffle=True),
        "Sequence_Interpro": load_data(modality_pair="Sequence_Interpro", config=config, batch_size=args.train_batch, device=args.device, shuffle=True),
        "Sequence_Ontology": load_data(modality_pair="Sequence_Ontology", config=config, batch_size=args.train_batch, device=args.device, shuffle=True),
    }

    
    val_dataloaders = {
        "Sequence_Structure": load_data(modality_pair="Sequence_Structure", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
        "Sequence_Text": load_data(modality_pair="Sequence_Text", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
        "Sequence_Interpro": load_data(modality_pair="Sequence_Interpro", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
        "Structure_Text": load_data(modality_pair="Structure_Text", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
        "Structure_Interpro": load_data(modality_pair="Structure_Interpro", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
        "Text_Interpro": load_data(modality_pair="Text_Interpro", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),

        "Sequence_Ontology": load_data(modality_pair="Sequence_Ontology", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
        "Structure_Ontology": load_data(modality_pair="Structure_Ontology", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
        "Text_Ontology": load_data(modality_pair="Text_Ontology", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True),
        "Interpro_Ontology": load_data(modality_pair="Interpro_Ontology", config=config, batch_size=args.valid_batch, device=args.device, shuffle=False, validation=True)
    }


    model = SeqBindPretrain(config=config, ontology=True).to(args.device)
    print(model)

    exit()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


    criteria = { 
        "Sequence_Structure": InfoNCELoss(logit_scale=1/0.07, device=args.device),
        "Sequence_Text": InfoNCELoss(logit_scale=1/0.05, device=args.device),
        "Sequence_Interpro": InfoNCELoss(logit_scale=1/0.06, device=args.device),
        "Sequence_Ontology": InfoNCELoss(logit_scale=1/0.04, device=args.device),
        "emergent_binding": InfoNCELoss(logit_scale=1/0.01, device=args.device),
        "diversity": DiversityLoss()
        }


    metrics = {
        "Similarity": {
            "Sequence_Structure": Similarity(logit_scale=1/0.07),
            "Sequence_Text": Similarity(logit_scale=1/0.05),
            "Sequence_Interpro": Similarity(logit_scale=1/0.06),
            "Sequence_Ontology": Similarity(logit_scale=1/0.04),
            "emergent_binding": Similarity(logit_scale=1/0.01),
        }
    }


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    #warmup_epochs = 5
    #warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / warmup_epochs)
    #scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_epochs])

    warmup_epochs = int(0.1 * args.epochs)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs-warmup_epochs, eta_min=1e-7)

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    
    ckp_dir = '/home/fbqc9/Workspace/MCLLM_DATA/DATA/saved_models/'
    ckp_file = ckp_dir + "pretrained_ontology_prostt5_new"
    

    if args.load_weights and os.path.exists(ckp_file):
        print("Loading model checkpoint @ {}".format(ckp_file))
        model, optimizer, lr_scheduler, current_epoch = load_ckp(filename=ckp_file, model=model, 
                                                                 optimizer=optimizer, lr_scheduler=lr_scheduler)
    else:
        current_epoch = 0
        min_val_loss = np.inf

    
    train(model=model, train_dataloader=train_dataloaders, val_dataloader=val_dataloaders,
          criteria=criteria, metrics= metrics, optimizer=optimizer, scheduler=scheduler, current_epoch=current_epoch)
    

    
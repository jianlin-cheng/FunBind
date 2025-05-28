import os
import re
import pandas as pd
import torch
from Metrics import PreRecF
from data_processing.extract_data import combine_modalities
from data_processing.utils import load_pickle, save_pickle
from models.model import  SeqBindClassifier
from utils import load_ckp, load_config, load_graph
from transformers import EsmTokenizer, T5Tokenizer, AutoTokenizer
from transformers import EsmModel, T5EncoderModel, AutoModel, AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch.nn.functional as F
from data_processing.dataset import CustomDataset, CustomDataCollator
from torch.utils.data import Dataset, DataLoader
import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:1"



def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config



class CustomDataCollator:
    def __init__(self, device=None):

        self.device = device or torch.device('cpu')
        self.modality_names = ['Sequence', 'Structure', 'Text', 'Interpro']

    def _move_to_device(self, input_ids):
        return torch.stack(input_ids, 0).to(self.device)


    def __call__(self, batch):

        proteins = [item[0] for item in batch]

        modalities = {}
        for idx, modality_name in enumerate(self.modality_names, start=1):
            modality_data = [item[idx] for item in batch]
            modalities[f'{modality_name}_modality'] = self._move_to_device(modality_data)
        
        modalities['Protein'] = proteins
        return modalities

            

class CustomDataset(Dataset):
    def __init__(self, config, data_path=None):

        self.config = config
        self.base_path = data_path or '/home/fbqc9/Workspace/MCLLM_DATA/DATA/data'

        train_valid_data = load_pickle(f"{self.base_path}/train_valid")
        self.data_list = train_valid_data["Validation"]
        self.data_list = list(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = torch.load(
            f'{self.base_path}/dataset_new/{self.data_list[idx]}.pt',
            weights_only=False
        )

        prot = sample['Protein']
        seq = sample['Sequence']['esm2_t48']
        struct = sample['Structure']['prost5']
        text = sample['Text']['llama2']
        interpro = sample['Interpro']['llama2']
        return (prot, seq, struct, text, interpro)


def load_model(ontology, model_name, device):
    config = load_config('config.yaml')['config1']
    model = SeqBindClassifier(config=config, go_ontology=ontology).to(device)
    ckp_dir = '/home/fbqc9/Workspace/MCLLM_DATA/DATA/saved_models/'
    ckp_file = ckp_dir + f"{ontology}_{model_name}.pt"
    print("Loading model checkpoint @ {}".format(ckp_file))
    loaded_model = load_ckp(filename=ckp_file, model=model, model_only=True, strict=True)
    return loaded_model


def fuse_predictions(predictions_dict, modality_weights, go_terms_list, go_graph, go_set):
    proteins = set()
    
    for modality in predictions_dict.values():
        proteins.update(modality.keys())

    combined_predictions = {}
    for protein in proteins:
        combined_predictions[protein] = {}
        for go_term in go_terms_list:
            weighted_sum = 0.0
            total_weight = 0.0
            for modality, modality_predictions in predictions_dict.items():
                if protein in modality_predictions:
                    weighted_sum += modality_predictions[protein][go_term] * modality_weights[modality]
                    total_weight += modality_weights[modality]
            if total_weight > 0:
                combined_predictions[protein][go_term] = weighted_sum / total_weight
            else:
                combined_predictions[protein][go_term] = 0.0


        # propagate terms
        for go_term, max_score in list(combined_predictions[protein].items()):
            if go_term in go_graph:
                parents = nx.descendants(go_graph, go_term).intersection(go_set)
                for parent in parents:
                    combined_predictions[protein][parent] = max(combined_predictions[protein].get(parent, 0), max_score)

    return combined_predictions


def generate_weight_combinations(step=0.1):
    combinations = {}
    scale = int(1 / step)

    for seq in range(0, scale + 1):
        for struct in range(0, scale + 1 - seq):
            for text in range(0, scale + 1 - seq - struct):
                interpro = scale - seq - struct - text
                if interpro < 0:
                    continue

                # Convert to decimals
                seq_w = round(seq * step, 1)
                struct_w = round(struct * step, 1)
                text_w = round(text * step, 1)
                interpro_w = round(interpro * step, 1)

                key = f"{seq_w}_{struct_w}_{text_w}_{interpro_w}"
                combinations[key] = (seq_w, struct_w, text_w, interpro_w)

    return combinations


def main():
    

    ontology = "BP"
    model_name = "unfrozen"
    batch_size = 7000
    device = "cuda:1"
    encoder =  {'Sequence': 'esm2_t48', 'Structure': 'prost5', 'Interpro': 'llama2', 'Text': 'llama2'}
    data_path = "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/dataset_new"  
    modalities_pred = ["Sequence", "Structure", "Text", "Interpro"]


    config = load_config('config.yaml')['config1']

    dataset = CustomDataset(config=config)
    collate_fxn = CustomDataCollator(device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fxn)

    labels = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{}_labels".format(ontology))
    info_acc = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{}_ia".format(ontology))

    metric = PreRecF(infor_accr=info_acc)

    model = load_model(ontology=ontology, model_name=model_name, device=device)
    model.eval()

    for batch in dataloader:
        proteins = batch['Protein']

        _, classification_outputs, _ = model(batch)

        

        _labels = [labels[protein] for protein in proteins]
        cls_labels = torch.tensor(_labels).float().to(device)


    consensus = {}
    combinations = generate_weight_combinations(step=0.1)
    for key, weights in combinations.items():
        consensus[key] = classification_outputs["Sequence_modality"] * weights[0] + classification_outputs["Structure_modality"] * weights[1] + classification_outputs["Text_modality"] * weights[2] + classification_outputs["Interpro_modality"] * weights[3]

    data = []
    for configuration, modality_output in consensus.items():
        res = metric.compute_scores(cls_labels, modality_output)
        seq, struct, text, interpro = map(float, configuration.split("_"))
        fscore = res['Weighted Fscore']
        data.append({
            "Sequence": seq,
            "Structure": struct,
            "Text": text,
            "Interpro": interpro,
            "Weighted Fscore": fscore
        })

    df = pd.DataFrame(data)


    top10 = df.sort_values(by="Weighted Fscore", ascending=False).head(10)
    print(top10)


    filtered_df = df[df["Structure"] > 0.0]
    top10 = filtered_df.sort_values(by="Weighted Fscore", ascending=False).head(10)
    print(top10)

    '''filtered_df = df[(df["Structure"] > 0.0) & (df["Interpro"] > 0.0)]
    top3 = filtered_df.sort_values(by="Weighted Fscore", ascending=False).head(3)
    print(top3)'''








if __name__ == "__main__":
    main()
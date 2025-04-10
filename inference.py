import os
import re
import torch
from Metrics import PreRecF
from data_processing.extract_data import combine_modalities
from data_processing.utils import load_pickle, save_pickle
from models.model import SeqBind, SeqBindClassifier
from utils import load_ckp, load_config, load_graph
from transformers import EsmTokenizer, T5Tokenizer, AutoTokenizer
from transformers import EsmModel, T5EncoderModel, AutoModel, AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from data_processing.dataset import CustomDataset, CustomDataCollator
from torch.utils.data import Dataset, DataLoader
import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:1"


class CustomDataCollator:
    def __init__(self, modality, device=None):

        self.modality = modality
        self.device = device or torch.device('cpu')
        self.modality_names = ['Sequence', 'Structure', 'Text', 'Interpro']

    def _move_to_device(self, input_ids):
        return torch.stack(input_ids, 0).to(self.device)


    def __call__(self, batch):

        proteins = [item[0] for item in batch]

        if self.modality == "Fused":
            modalities = {}
            for idx, modality_name in enumerate(self.modality_names, start=1):
                modality_data = [item[idx] for item in batch]
                modalities[f'{modality_name}_modality'] = self._move_to_device(modality_data)
            
            modalities['Protein'] = proteins
            return modalities
        else:
            modality_data = [item[1] for item in batch]
            return {
                'Protein': proteins,
                f'{self.modality}_modality': self._move_to_device(modality_data)
            }
            


class InferenceDataset(Dataset):
    def __init__(self, modality,  data_path=None, data_list=None):

        self.modality = modality
        self.base_path = data_path or '/home/fbqc9/Workspace/MCLLM_DATA/DATA/test'
        self.data_list = data_list

        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = torch.load(
            f'{self.base_path}/{self.data_list[idx]}.pt',
            weights_only=False
        )

        if self.modality == "Fused":
            prot = sample['Protein']
            seq = sample['Sequence']['esm2_t48']
            struct = sample['Structure']
            text = sample['Text']
            interpro = sample['Interpro']
            return (prot, seq, struct, text, interpro)
        else:
            prot = sample['Protein']
            if self.modality == "Sequence":
                modality = sample[self.modality]['esm2_t48']
            else:
                modality = sample.get(self.modality)
            return (prot, modality)
            


def get_embeddings(model, data, modality):

    data = str(data)

    if modality == "Structure":
        data = data.lower()
        data = re.sub(r"[UZOB]", "X", data)
        data = " ".join(list(data))
        data = "<fold2AA>" + " " + data


    if modality == "Text" or modality == "Interpro" or modality == "Ontology":
        inputs = model.config.tokenizer(data, return_tensors="pt", max_length=1024, padding='max_length', truncation=True)
    else:
        inputs = model.config.tokenizer(data, return_tensors="pt")

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    outputs = torch.mean(outputs.last_hidden_state, dim=1).cpu()

    return outputs


def get_model(model_name):
    model_map = {
        "esm2_t48": ('facebook/esm2_t48_15B_UR50D', EsmTokenizer, EsmModel),
        "prost5":   ('Rostlab/ProstT5', T5Tokenizer, T5EncoderModel),
        "llama2":   ('meta-llama/Llama-2-7b-hf', AutoTokenizer, AutoModel),
    }

    model_info = model_map.get(model_name)
    if model_info is None:
        raise ValueError(f"Model {model_name} is not recognized.")
    
    model_path, tokenizer_class, model_class = model_info

    if 'prost5' in model_name:
        tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=False)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_path)
    
    model = model_class.from_pretrained(model_path)
    model.config.tokenizer = tokenizer
    
    if "llama" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model.tokenizer = tokenizer
    return model.to(device)


def get_modality_list(data):

    modality_list = {
        'Sequence': set(),
        'Structure': set(),
        'Text': set(),
        'Interpro': set()
    }
    
    for protein, protein_data in data.items():
        for modality in modality_list:
            if modality in protein_data:
                modality_list[modality].add(protein)
    
    all_modalities = set.intersection(*modality_list.values())
    modality_list['Combined'] = all_modalities

    return modality_list


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


def main():
    

    ontology = "BP"
    model_name = "unfrozen"
    num_batches = 120
    encoder =  {'Sequence': 'esm2_t48', 'Structure': 'prost5', 'Interpro': 'llama2', 'Text': 'llama2'}
    data_path = "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/dataset_new"  
    modalities_pred = ["Sequence", "Structure", "Text", "Interpro"]


    thresholds = {
        "BP": [0.3, 0.1, 0.5, 0.2],
        "CC": [0.3, 0.1, 0.4, 0.2],
        "MF": [0.3, 0.1, 0.3, 0.3]
    }

    FUNC_DICT = {
        'CC': 'GO:0005575',
        'MF': 'GO:0003674',
        'BP': 'GO:0008150'
        }


    go_terms_list = load_pickle(f"/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{ontology}_terms")


    sequence_fasta = "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/raw/sequence.fasta"
    structure_fasta = "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/raw/structure.fasta"
    text_data = "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/raw/text.txt"
    interpro_data = "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/raw/interpro2.txt"

    go_graph = load_graph(graph_pth="/home/fbqc9/Workspace/MCLLM/evaluation/go-basic.obo")
    go_set = nx.ancestors(go_graph, FUNC_DICT[ontology])

    data = combine_modalities(sequence_data=sequence_fasta, 
                              structure_data=structure_fasta,
                              textual_data=text_data, 
                              interpro_data=interpro_data)


    modality_list = get_modality_list(data)
    test_proteins = load_pickle("/home/fbqc9/Workspace/MCLLM/evaluation/proteins")[ontology].difference(set(['P0DXI8', 'P0DXI6']))


    print(len(list(modality_list['Interpro'].intersection(test_proteins))))


    generated = os.listdir("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/dataset_new")
    generated = set([i.split(".")[0] for i in generated])

    protein_list = list(set(data.keys()) - generated)
    print(f"Generated: {len(generated)}, Remaining: {len(protein_list)}")


    
    ##### Generate data 
    if len(protein_list) > 0:
        encoding_models = {key: get_model(value) for key, value in encoder.items()}

    for prot_name in protein_list:

        try:

            print(prot_name)
        
            sequence = data[prot_name].get('Sequence', None)
            structure = data[prot_name].get('Structure', None)
            text = data[prot_name].get('Text', None)
            interpro = data[prot_name].get('Interpro', None)


            protein_data = {"Protein": prot_name}

            if sequence:
                sequence_embedding = get_embeddings(encoding_models["Sequence"], data[prot_name]['Sequence'], "Sequence")
                protein_data["Sequence"] = sequence_embedding
                
            if structure:
                structures_embedding = get_embeddings(encoding_models["Structure"], data[prot_name]['Structure'], "Structure")
                protein_data["Structure"] = structures_embedding

            if text:
                texts_embedding = get_embeddings(encoding_models["Text"], data[prot_name]['Text'], "Text")
                protein_data["Text"] = texts_embedding

            if interpro:
                interpros_embedding = get_embeddings(encoding_models["Interpro"], data[prot_name]['Interpro'], "Interpro")
                protein_data["Interpro"] = interpros_embedding

            # torch.save(protein_data, f"/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/dataset_new/{prot_name}.pt")
        except torch.cuda.OutOfMemoryError:
            pass


    # load model and predict
    model = load_model(ontology=ontology, model_name=model_name, device=device)
    model.eval()


    predictions_dict = {
        "Sequence" : {},
        "Structure": {},
        "Text": {}, 
        "Interpro": {},
        "Combined": {}
    }

    with torch.no_grad():
        for mod in modalities_pred:
            dataset = InferenceDataset(
                            modality=mod, 
                            data_path=data_path, 
                            data_list=list(modality_list[mod].intersection(test_proteins))
                            )
            
            print(mod, len(dataset))
            collator = CustomDataCollator(modality=mod, device=device)
            dataloader = DataLoader(dataset, batch_size=num_batches, shuffle=False, collate_fn=collator)

            for batch in dataloader:
                proteins = batch['Protein']
                _, classification_outputs, _ = model(batch)
                predictions = classification_outputs[f'{mod}_modality']

                for i, protein in enumerate(proteins):
                    protein_scores  = {}

                    for j, go_term in enumerate(go_terms_list):
                        protein_scores[go_term] = predictions[i, j].item()
                    
                    # Propagate scores to parent terms
                    for go_term, max_score in list(protein_scores.items()):
                        if go_term in go_graph:
                            parents = nx.descendants(go_graph, go_term).intersection(go_set)
                            for parent in parents:
                                 protein_scores[parent] = max(protein_scores.get(parent, 0), max_score)

                    predictions_dict[mod][protein] = protein_scores


        predictions_dict["Combined"] = fuse_predictions(predictions_dict, 
                                modality_weights={"Sequence": thresholds[ontology][0], 
                                                  "Structure": thresholds[ontology][1],
                                                  "Text": thresholds[ontology][2], 
                                                  "Interpro": thresholds[ontology][3]},
                                go_terms_list=go_terms_list,
                                go_graph=go_graph,
                                go_set=go_set)

    
    for mod, protein_predictions in predictions_dict.items():
        with open(f"/home/fbqc9/Workspace/MCLLM/evaluation/ablation/{ontology}/{mod}_{model_name}.tsv", "w") as f:
            for protein, go_term_scores in protein_predictions.items():
                for go_term, score in go_term_scores.items():
                    if score >= 0.01:
                        f.write(f"{protein}\t{go_term}\t{score:.4f}\n")



if __name__ == "__main__":
    main()
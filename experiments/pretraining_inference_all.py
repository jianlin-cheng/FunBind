from collections import Counter, defaultdict
import itertools
import math
import os
import random
import re
import numpy as np
import pandas as pd
import torch
from transformers import EsmTokenizer, T5Tokenizer, AutoTokenizer
from transformers import EsmModel, T5EncoderModel, AutoModel, AutoTokenizer
from Metrics import Retrieve_MRR, Retrive_at_k
from data_processing.utils import load_pickle
from models.model import SeqBindPretrain
from utils import load_ckp, load_config, load_graph
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import networkx as nx
from matplotlib.lines import Line2D

def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def propagate(go_graph, terms):
    res = set()
    for term in terms:
        try:
            tmp = nx.descendants(go_graph, term).union({term})
            res.update(tmp)
        except nx.NetworkXError:
            pass
    return res

def collect_data(ontology=None):
    go_graph = load_graph()

    ontologies = ["CC", "MF", "BP"]

    if ontology:
        ontologies = [ontology]
    
    res = {}
    for ontology in ontologies:
        pth = f"/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{ontology}_groundtruth"
        data = load_pickle(pth)
        training_terms = set([term for go_terms in data.values() for term in go_terms])

        data_test = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/test_data")[ontology]


        for key, value in data_test.items():
            tmp = propagate(go_graph, value)
            new_terms = list(tmp.difference(training_terms).difference(set(['GO:0160228', 'GO:0160221', 'GO:0120532'])))
            new_terms.sort()
            if len(new_terms) > 0:
                if not key in res:
                    res[key] = []
                for i in new_terms:
                    if len(res[key]) < 1:
                        res[key].append((i, ontology))

    return res


def create_protein_groups(protein_dict, min_group_size=10, max_group_size=12):

    proteins = list(protein_dict.keys())
    total_proteins = len(proteins)

    ideal_group_size = (min_group_size + max_group_size) // 2
    num_groups = total_proteins // ideal_group_size

    random.shuffle(proteins)

    base_size = total_proteins // num_groups
    surplus = total_proteins % num_groups

    grouped_proteins = []
    start = 0
    for i in range(num_groups):
        group_size = base_size + (1 if surplus > 0 else 0)
        grouped_proteins.append(proteins[start:start + group_size])
        start += group_size
        surplus -= 1

    result = []
    for group in grouped_proteins:
        res = []
        for protein in group:
            term = tuple(go[0] for go in protein_dict[protein])
            res.append((protein, term))
        result.append(res)
        
    return result




def process_ontology(go_graph, ontology_list):
    ontology_namespace = {
        'molecular_function': 'Molecular Function',
        'biological_process': 'Biological Process',
        'cellular_component': 'Cellular Component',
    }

    grouped_results = []

    for i, ontology_group in enumerate(ontology_list):
        namespace_groups = {'Cellular Component': [], 'Molecular Function': [], 'Biological Process': []}

        for ontology in ontology_group:
            namespace = ontology_namespace[go_graph.nodes[ontology]['namespace']]
            name = go_graph.nodes[ontology]['name']
            definition = go_graph.nodes[ontology]['def']
            cleaned_definition = re.sub(r'\[.*?\]', '', definition).replace('"', '')

            namespace_groups[namespace].append(f"{name}: {cleaned_definition}")

        formatted_string = ""
        for namespace, descriptions in namespace_groups.items():
            if descriptions:
                namespace_section = f"{namespace}:: " + "; ".join(descriptions)
                formatted_string += namespace_section
        
        grouped_results.append(formatted_string.strip())

    return grouped_results


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
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path)
    
    if "llama" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model.tokenizer = tokenizer
    return model.to(device)


def get_embeddings(model, data, modality):

    if modality == "Sequence":
        data = str(data)
    if modality == "Structure":
        data = str(data).lower()
        data = re.sub(r"[UZOB]", "X", data)
        data = " ".join(list(data))
        data = "<fold2AA>" + " " + data

    if modality == "Text" or modality == "Interpro" or modality == "Ontology":
        inputs = model.tokenizer(data, return_tensors="pt", max_length=1024, padding='max_length', truncation=True)
    else:
        inputs = model.tokenizer(data, return_tensors="pt")

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    outputs = torch.mean(outputs.last_hidden_state, dim=1)
    return outputs


def load_model():
    config = load_config('config.yaml')['config1']
    model = SeqBindPretrain(config=config).to(device)
    ckp_dir = '/home/fbqc9/Workspace/MCLLM_DATA/DATA/saved_models/'
    ckp_file = ckp_dir + "pretrained_ontology.pt"
    print("Loading model checkpoint @ {}".format(ckp_file))
    load_model = load_ckp(filename=ckp_file, model=model, model_only=True)
    return load_model


def get_same_position_indices(lst):
    index_map = defaultdict(list)
    for i, val in enumerate(lst):
        index_map[val].append(i)
    result = [index_map[val] for val in lst]
    return result


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



def compute_similarity(group, model):

    go_graph = load_graph()
    model_map = {"Sequence": "esm2_t48", "Structure": "prost5", "Text": "llama2", "Interpro": "llama2"}

    protein_list, term_list = zip(*group) 
    protein_list = list(protein_list)
    terms = [i[0] for i in term_list]

    all_embeddings = {}

    for modality in model_map:
        tmp_embeddings = []
        for protein in protein_list:
            embedding = torch.load(f'/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/dataset_new/{protein}.pt')
            if modality == 'Sequence':
                tmp_embeddings.append(embedding[modality][model_map[modality]])
            else:
                tmp_embeddings.append(embedding[modality])

        all_embeddings[f"{modality}"] = torch.stack(tmp_embeddings, dim=0).to(device)

    
    ontology_text = process_ontology(go_graph, list(term_list))
    ontology_embeddings = []
    for batch in range(0, len(ontology_text), 32):
        batch_text = ontology_text[batch:batch + 32]
        ontology_embeddings.append(get_embeddings(get_model("llama2"), batch_text, "Ontology"))
    
    all_embeddings["Ontology"] = torch.cat(ontology_embeddings, dim=0).to(device)


    outputs = {}
    with torch.no_grad():
        for modality, embeddings in all_embeddings.items():
            tmp, _ = model.encode_modality(modality=f'{modality}_modality', value=embeddings.squeeze(1))
            outputs[modality] = F.normalize(tmp, dim=-1)


    results_dict = {}
    for key, value in outputs.items():
        if key == "Ontology":
            continue
        else:
            ret_1 = retrieve_at_1(modality1_features=value, modality2_features=outputs["Ontology"], groundtruth_all_indices=get_same_position_indices(terms))
            ret_3 = retrieve_at_3(modality1_features=value, modality2_features=outputs["Ontology"], groundtruth_all_indices=get_same_position_indices(terms))
            ret_5 = retrieve_at_5(modality1_features=value, modality2_features=outputs["Ontology"], groundtruth_all_indices=get_same_position_indices(terms))
            ret_mrr = retrieve_mrr(modality1_features=value, modality2_features=outputs["Ontology"], groundtruth_all_indices=get_same_position_indices(terms))

            results_dict[key] = (ret_1.item(), ret_3.item(), ret_5.item(), ret_mrr.item())

    return results_dict


def plot(datasets):
    metrics = ('Ret@1', 'Ret@3', 'Ret@5', 'MRR')
    ontologies = ['CC', 'MF', 'BP', 'ALL']
    colors = ["#143F75", "#A2C54C", "#F79646", "#7030A0"]
    modalities = ['Sequence', 'Structure', 'Text', 'Interpro']
    ont_map = {'CC': 'Cellular Component', 'MF': 'Molecular Function', 'BP': 'Biological Process'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    width = 0.2
    
    for i, (ont, data) in enumerate(datasets.items()):
        ax = axes[i]
        x = np.arange(len(metrics))

        for j, (attribute, measurement) in enumerate(data.items()):
            ax.bar(x + j * width, measurement, width, color=colors[j])


        ax.set_xticks(x + width * (len(ontologies) - 1) / 2)
        ax.set_xticklabels(metrics, fontsize=18)
        ax.set_ylabel('Score', fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_title(f"{ont_map.get(ont, 'All')}", fontsize=18)


    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(modalities))]
    fig.legend(handles, modalities, loc='lower center', ncol=len(modalities), bbox_to_anchor=(0.5, 0.005), fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f"pretraining_plots/{'retrival_no_grouping_old'}.png")


def run_experiment(ontology=None):

    datalist = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/test_data_modality")
    datalist = set(datalist["Structure"]).intersection(set(datalist["Text"])).intersection(set(datalist["Interpro"]))

    data = collect_data(ontology=ontology)
    print(f"Number of proteins is {len(data)}")



    print(len(data))




    exit()

    data = {key: data[key] for key in datalist if key in data}

    print(f"Number of proteins is {len(data)}")

    
    model = load_model()
    model.eval()

    # One group
    groups = create_protein_groups(data, min_group_size=len(data), max_group_size=len(data))
    print(f"Number of groups is {len(groups)}")


    return compute_similarity(group=groups[0], model=model)



set_seeds(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda:1"

retrieve_at_1 = Retrive_at_k(k=1)
retrieve_at_3 = Retrive_at_k(k=3)
retrieve_at_5 = Retrive_at_k(k=5)
retrieve_mrr = Retrieve_MRR()

ontologies = ["CC", "MF", "BP", "All"]

data = {}

for i in ontologies:
    if i == "All":
        ontology = None
    else:
        ontology = i

    
    print("Ontology: {}".format(ontology))
    data[i] = run_experiment(ontology=ontology)

plot(data)
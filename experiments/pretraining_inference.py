from collections import Counter, defaultdict
import itertools
import math
import os
import random
import re
import numpy as np
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

    # if ontology is None, all ontologies
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
            new_terms = tmp.difference(training_terms).difference(set(['GO:0160228', 'GO:0160221', 'GO:0120532']))
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
    model = SeqBindPretrain(config=config, ontology=True).to(device)
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


def compute_similarity(group, modality1, model):

    go_graph = load_graph()
    model_map = {"Sequence": "esm2_t48", "Structure": "prost5", "Text": "llama2", "Interpro": "llama2"}

    protein_list, term_list = zip(*group) 
    protein_list = list(protein_list)
    terms = [i[0] for i in term_list]

    modality1_embeddings = []
    for protein in protein_list:
        embedding = torch.load(f'/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/dataset_new/{protein}.pt')
        if modality1 == 'Sequence':
            modality1_embeddings.append(embedding[modality1][model_map[modality1]])
        else:
            modality1_embeddings.append(embedding[modality1])
    modality1_embeddings = torch.stack(modality1_embeddings, dim=0).to(device)
    
    ontology_text = process_ontology(go_graph, list(term_list))
    ontology_embeddings = get_embeddings(get_model("llama2"), ontology_text, "Ontology")


    with torch.no_grad():
        mod1_features, _ = model.encode_modality(modality=f'{modality1}_modality', value=modality1_embeddings.squeeze(1))
        ont_features, _ = model.encode_modality(modality='Ontology_modality', value=ontology_embeddings)

    mod1_features = F.normalize(mod1_features, dim=-1)
    ont_features = F.normalize(ont_features, dim=-1)


    ret_1 = retrieve_at_1(modality1_features=mod1_features, modality2_features=ont_features, groundtruth_all_indices=get_same_position_indices(terms))
    ret_3 = retrieve_at_3(modality1_features=mod1_features, modality2_features=ont_features, groundtruth_all_indices=get_same_position_indices(terms))
    ret_5 = retrieve_at_5(modality1_features=mod1_features, modality2_features=ont_features, groundtruth_all_indices=get_same_position_indices(terms))
    ret_mrr = retrieve_mrr(modality1_features=mod1_features, modality2_features=ont_features, groundtruth_all_indices=get_same_position_indices(terms))


    similarity = (50.0 * mod1_features @ ont_features.T).softmax(dim=-1)

    # x_labels = [', '.join(tup) for tup in term_list]
    return similarity.cpu(), terms, protein_list, {"Ret@1": ret_1.item(), "Ret@3": ret_3.item(), "Ret@5": ret_5.item(), "MRR": ret_mrr.item()}


# {'Ret@1': 0.6719697117805481, 'Ret@3': 0.9025252742899789, 'Ret@5': 0.9386363807651732, 'MRR': 0.7919113006856706}
# {'Ret@1': 0.6848485006226434, 'Ret@3': 0.8929293155670166, 'Ret@5': 0.9439394093222089, 'MRR': 0.7941735088825226}
# {'Ret@1': 0.6506493657827377, 'Ret@3': 0.8820346508707319, 'Ret@5': 0.9379004461424691, 'MRR': 0.7764759353228978}
# {'Ret@1': 0.652962297626904,  'Ret@3': 0.8921645198549543, 'Ret@5': 0.9425108347620282, 'MRR': 0.7845069084848676}


def run_experiment(modality="Sequence", ontology=None):

    datalist = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/test_data_modality")

    data = collect_data(ontology=ontology)

    print(len(data))
    exit()

    if modality != 'Sequence':
        data = {key: data[key] for key in datalist[modality] if key in data}


    model = load_model()
    model.eval()

    groups = create_protein_groups(data)
    print(f"Number of groups is {len(groups)}")


    step_size = 12
    num_groups = len(groups)
    all_metrics = {"Ret@1": [],"Ret@3": [], "Ret@5": [], "MRR": []}
    
    for start_idx in range(0, num_groups, step_size):

        end_idx = min(start_idx + step_size, num_groups)
        subgroup = groups[start_idx:end_idx]

        print(f"Processing groups {start_idx + 1} to {end_idx}...")

        # rows, cols = 3, 2
        # rows, cols = 4, 4
        # rows, cols = 1, 3
        rows, cols = 4, 3

        # fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(24, 24))
        # fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(24, 18))
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(24, 24))
        # fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 6))

        for idx, group in enumerate(subgroup):

            sim, Xs, Ys, metrics = compute_similarity(group=group, modality1=modality, model=model)
            
            all_metrics["Ret@1"].append(metrics["Ret@1"])
            all_metrics["Ret@3"].append(metrics["Ret@3"])
            all_metrics["Ret@5"].append(metrics["Ret@5"])
            all_metrics["MRR"].append(metrics["MRR"])

            i, j = divmod(idx, cols)

            im = sns.heatmap(sim, xticklabels=Xs, yticklabels=Ys, cmap='YlOrRd', ax=axes[i, j])
            axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor", fontsize=16)
            axes[i, j].set_yticklabels(axes[i, j].get_yticklabels(), rotation=45, ha='right', rotation_mode="anchor", fontsize=16)
            axes[i, j].set_title(f"Group {idx + 1}")

            '''im = sns.heatmap(sim, xticklabels=Xs, yticklabels=Ys, cmap='YlOrRd', ax=axes[j])
            axes[j].set_xticklabels(axes[j].get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor", fontsize=16)
            axes[j].set_yticklabels(axes[j].get_yticklabels(), rotation=45, ha='right', rotation_mode="anchor", fontsize=16)
            axes[j].set_title(f"Group {idx + 1}")'''

        if ontology:
            fig.suptitle(f"{modality} -- {ontology}", fontsize=24, y=0.99)
        else:
            fig.suptitle(f"{modality}", fontsize=24, y=0.99)

        fig.supxlabel("GO Terms", fontsize=20)
        fig.supylabel("Proteins", fontsize=20)

        plt.tight_layout()

        if ontology:
            plt.savefig(f"pretraining_plots2/{ontology}_{modality}_{start_idx + 1}_{end_idx}.png")
        else:
            plt.savefig(f"pretraining_plots2/{modality}_{start_idx + 1}_{end_idx}.png")

    all_metrics = {key: sum(value) / len(value) for key, value in all_metrics.items()}
    return all_metrics


def run_n_times(ontology, modality, n=10):

    all_runs_metrics = defaultdict(list)
    for i in range(n):
        print("Run {}...".format(i + 1))
        metrics = run_experiment(modality=modality, ontology=ontology)
        for key, value in metrics.items():
            print(key, value)
            all_runs_metrics[key].append(value)

     # Calculate and print mean and standard deviation in a table format
    print("\nResults (Mean +/- SD):")
    print("-" * 30)
    print("{:<10} {:<15}".format("Metric", "Value"))
    print("-" * 30)

    for key, values in all_runs_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print("{:<10} {:.4f} +/- {:.4f}".format(key, mean, std))


set_seeds(42)

modality = "Text"
ontology = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cuda:0"

retrieve_at_1 = Retrive_at_k(k=1)
retrieve_at_3 = Retrive_at_k(k=3)
retrieve_at_5 = Retrive_at_k(k=5)
retrieve_mrr = Retrieve_MRR()

print("Ontology: {}, Modality: {}".format(ontology, modality))
run_n_times(ontology=ontology, modality=modality, n=10)

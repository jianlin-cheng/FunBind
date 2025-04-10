from collections import Counter, defaultdict
import random
import numpy as np
import obonet
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import EsmTokenizer, BertTokenizer, T5Tokenizer, AutoTokenizer
from transformers import EsmModel, BertModel, T5Model, T5EncoderModel, AutoModel, AutoTokenizer
import os, re
from utils import load_pickle, save_pickle
import networkx as nx


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cuda:1"
# device = "cpu"

def get_embeddings(model, data):
    if model.config._name_or_path == "dmis-lab/biobert-large-cased-v1.1-mnli":
        inputs = model.config.tokenizer(data, return_tensors="pt", max_length=512, padding='max_length', truncation=True)

    elif "llama" in model.config._name_or_path.lower():
        inputs = model.config.tokenizer(data, return_tensors="pt", max_length=1024, padding='max_length', truncation=True)

    else:
        inputs = model.config.tokenizer(data, return_tensors="pt")

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    outputs = torch.mean(outputs.last_hidden_state, dim=1).cpu()
    return outputs


def preprocess_prost5(data, seq_struct):

    if seq_struct == "Structure":
        data = data.lower()
        data = re.sub(r"[UZOB]", "X", data)
        data = " ".join(list(data))
        data = "<fold2AA>" + " " + data

    elif seq_struct == "Sequence":
        data = re.sub(r"[UZOB]", "X", data)
        data = " ".join(list(data))
        data = "<AA2fold>" + " " + data

    return data
        

def get_model(model_name):
    model_map = {
        "esm2_t36": ('facebook/esm2_t36_3B_UR50D', EsmTokenizer, EsmModel),
        "esm2_t48": ('facebook/esm2_t48_15B_UR50D', EsmTokenizer, EsmModel),
        "prost5":   ('Rostlab/ProstT5', T5Tokenizer, T5EncoderModel),
        "biobert":  ('dmis-lab/biobert-large-cased-v1.1-mnli', BertTokenizer, BertModel),
        "llama2":   ('meta-llama/Llama-2-7b-hf', AutoTokenizer, AutoModel),
        "llama3":   ('meta-llama/Meta-Llama-3-8B', AutoTokenizer, AutoModel),
        "llama3.1": ('meta-llama/Meta-Llama-3.1-8B', AutoTokenizer, AutoModel),
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

    return model.to(device)


def generate_data(des_dir):

    raw_data = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/raw/combined_data")
    
    
    generated_data = set([i.split(".")[0] for i in os.listdir(des_dir)])
    remaining_data = set(raw_data.keys()).difference(generated_data)
    

    print(f"All Data: {len(raw_data)}, Generated Data: {len(generated_data)},\
           Remaining Data: {len(remaining_data)}")


    # Pre-load all models
    models = {
        'esm2_t36': get_model("esm2_t36"),
        'esm2_t48': get_model("esm2_t48"),
        'prost5': get_model("prost5"),
        'biobert': get_model("biobert"),
        'llama2': get_model("llama2"),
        'llama3': get_model("llama3"),
        'llama31': get_model("llama3.1")
    }


    for protein in remaining_data:

        try:

            embeddings = {'Protein': protein}

            features = raw_data[protein]

            # Handle "Sequence"
            if "Sequence" in features:
                embeddings['Sequence'] = {}
                sequence = str(features['Sequence'])
                seq_prost5 = preprocess_prost5(sequence, seq_struct="Sequence")
                embeddings['Sequence']['esm2_t36'] = get_embeddings(models['esm2_t36'], sequence)
                embeddings['Sequence']['esm2_t48'] = get_embeddings(models['esm2_t48'], sequence)
                embeddings['Sequence']['prost5'] = get_embeddings(models['prost5'], seq_prost5)

                
            
            if not features['Structure'] == None:
                embeddings['Structure'] = {}
                structure = preprocess_prost5(str(features['Structure']), seq_struct="Structure")
                embeddings['Structure']['prost5'] = get_embeddings(models['prost5'], structure)

            if not features['Text'] == None:
                embeddings['Text'] = {}
                text = str(features['Text'])
                embeddings['Text']['biobert'] = get_embeddings(models['biobert'], text)
                embeddings['Text']['llama2'] = get_embeddings(models['llama2'], text)
                embeddings['Text']['llama3'] = get_embeddings(models['llama3'], text)
                embeddings['Text']['llama31'] = get_embeddings(models['llama31'], text)
            
            if not features['Interpro'] == None:
                embeddings['Interpro'] = {}
                interpro = str(features['Interpro'])
                embeddings['Interpro']['biobert'] = get_embeddings(models['biobert'], interpro)
                embeddings['Interpro']['llama2'] = get_embeddings(models['llama2'], interpro)
                embeddings['Interpro']['llama3'] = get_embeddings(models['llama3'], interpro)
                embeddings['Interpro']['llama31'] = get_embeddings(models['llama31'], interpro)
                

            print("Saving protein {}".format(protein))
            print(embeddings)

            for i in embeddings['Sequence']:
                print(i, embeddings['Sequence'][i].shape)
            torch.save(embeddings, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/dataset_new/{}.pt".format(protein))  

        except Exception as e:
            print(e)
# generate_data(des_dir="/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/dataset_new")




def load_graph():
    go_graph = obonet.read_obo(open("/home/fbqc9/Workspace/MCLLM_DATA/DATA/cafa5/Train/go-basic.obo", 'r'))

    accepted_edges = set()
    unaccepted_edges = set()

    for edge in go_graph.edges:
        if edge[2] == 'is_a' or edge[2] == 'part_of':
            accepted_edges.add(edge)
        else:
            unaccepted_edges.add(edge)
    go_graph.remove_edges_from(unaccepted_edges)
    return go_graph


def create_protein2go(terms, ont):
    res_dict = defaultdict(set)

    for index, row in terms.iterrows():
        entry_id, term, aspect = row.iloc[0], row.iloc[1], row.iloc[2]

        if aspect == ont:
            res_dict[entry_id].add(term)

    return res_dict


def is_parent(graph, source, target):
    return nx.has_path(graph, source, target)


def generate_labels(ontology, cut_off=30):

    print(f"Generating for {ontology}")

    exclude_terms = {"GO:0005575", "GO:0003674", "GO:0008150"}

    ia = pd.read_csv("/home/fbqc9/Workspace/MCLLM_DATA/DATA/cafa5/IA.txt", sep='\t', header=None)
    go_dict = ia.set_index(0)[1].astype(float).to_dict()
    terms = pd.read_csv("/home/fbqc9/Workspace/MCLLM_DATA/DATA/cafa5/Train/train_terms.tsv", sep='\t')

    ont_dict = create_protein2go(terms, f"{ontology}O")
    go_term_counts = Counter(term for go_terms in ont_dict.values() for term in go_terms)
    filtered_go_terms = sorted([term for term, count in go_term_counts.items() if term not in exclude_terms and count > cut_off])
    print(len(filtered_go_terms))
    
    ont_ia = [go_dict[i] for i in filtered_go_terms]

    output = {}
    for protein, go_terms in ont_dict.items():
        binary_vector = [1 if term in go_terms else 0 for term in filtered_go_terms]
        output[protein] = binary_vector

    save_pickle(ont_dict, f"/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{ontology}_groundtruth")
    save_pickle(output, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{}_labels".format(ontology))
    save_pickle(ont_ia, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{}_ia".format(ontology))
    save_pickle(filtered_go_terms, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{}_terms".format(ontology))

generate_labels(ontology="MF", cut_off=20)

exit()



def generate_train_validation():

    random.seed(42)

    structure_list = set()
    text_list = set()
    interpro_list = set()
    sequence_list = set()
    data = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/raw/combined_data")


    # Has all modalities present
    for protein_id, modalities in data.items():
        if modalities["Structure"]:
            structure_list.add(protein_id)
        if modalities["Text"]:
            text_list.add(protein_id)
        if modalities["Interpro"]:
            interpro_list.add(protein_id)
        sequence_list.add(protein_id)


    #ab = structure_list.union(text_list).union(interpro_list)
    # print(len(sequence_list), len(structure_list), len(text_list), len(interpro_list), len(ab))


    # Has annotations in all ontology
    cc = set(load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/CC_labels").keys())
    mf = set(load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/MF_labels").keys())
    bp = set(load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/BP_labels").keys())

    print("len cc", len(cc), "len mf", len(mf), "len bp", len(bp))
    print("len structure", len(structure_list), "len text", len(text_list), "len interpro", len(interpro_list))
    print("len sequence", len(structure_list.union(text_list).union(interpro_list)))


    # pick validation from here
    x = cc.intersection(mf).intersection(bp)
    y = structure_list.intersection(text_list).intersection(interpro_list)
    z = list(y.intersection(x))
    print(len(z))
    # len cc 92912 len mf 78637 len bp 92210
    # len structure 125021 len text 110425 len interpro 126861
    validation_list =  set(random.sample(z, 7000))

    structure_list = structure_list.difference(validation_list)
    text_list = text_list.difference(validation_list)
    interpro_list = interpro_list.difference(validation_list)
    sequence_list = sequence_list.difference(validation_list)
    meta = structure_list.intersection(text_list).intersection(interpro_list)



    train_valid = {}
    train_valid['Structure'] = structure_list
    train_valid['Text'] = text_list
    train_valid['Interpro'] = interpro_list
    train_valid['Sequence'] = sequence_list
    train_valid['Validation'] = validation_list
    train_valid['meta'] = meta

    save_pickle(train_valid, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/train_valid")

# generate_train_validation()




def create_ontology_data():

    go_graph = load_graph()

    def get_terminals(terms):
        terminals = set()  # Initialize an empty set to store terminal terms
        
        # Iterate over the provided terms
        for term in terms:
            # Get all parents of the GO term (invert tree logic: check parents instead of children)
            parents = list(go_graph.predecessors(term))
            parent_terms_in_set = [parent for parent in parents if parent in terms]  # Check if parents are in the provided set
            
            # If no parent terms are in the set, it's a terminal (leaf node)
            if len(parent_terms_in_set) == 0:
                terminals.add(term)
        
        return terminals

    
    onts = ['CC', 'MF', 'BP']
    results = {}

    for ont in onts:

        if ont == "CC":
            ontology_name = "Cellular Component"
        elif ont == "MF":
            ontology_name = "Molecular Funtion"
        elif ont == "BP":
            ontology_name = "Biological Process"


        ont_dict = load_pickle(f"/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{ont}_groundtruth")

        for protein in ont_dict:

            if protein not in results:
                results[protein] = ""

            terminals = get_terminals(ont_dict[protein])

            if terminals:
                results[protein] += f"{ontology_name}::"


            for terminal in terminals:

                definition = go_graph.nodes[terminal]['def']
                cleaned_definition = re.sub(r'\[.*?\]', '', definition)
                cleaned_definition = re.sub(r'"', '', cleaned_definition) 

                results[protein] += "; " + go_graph.nodes[terminal]['name'] + ": " + cleaned_definition

            results[protein] = re.sub(r"::;", "::", results[protein])

    save_pickle(results, f'/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/ontology_list_{ont}')

# create_ontology_data()
 


# encode_ontology()
def get_batch_embeddings(model, batch_data):
    embeddings = model.generate_embeddings(batch_data)
    return embeddings

def encode_ontology_batch():
    input_file = '/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/ontology_list'
    output_dir = '/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/ontology_data'
    batch_size = 24

    x = load_pickle(input_file)


    generated_data = {i.split(".")[0] for i in os.listdir(output_dir)}
    remaining_data = sorted(set(x.keys()) - generated_data)


    print(len(remaining_data))

    model = get_model("llama2")

    # Process remaining data in batches
    for i in range(0, len(remaining_data), batch_size):
        batch_proteins = remaining_data[i:i + batch_size]
        
        batch_data = [x[protein] for protein in batch_proteins]


        inputs = model.config.tokenizer(batch_data, return_tensors="pt", max_length=1024, padding='max_length', truncation=True)

        inputs = {key: value.to(device) for key, value in inputs.items()}
        


        with torch.no_grad():
            outputs = model(**inputs)


        outputs = torch.mean(outputs.last_hidden_state, dim=1).cpu()


        for protein, embedding in zip(batch_proteins, outputs):
            output_path = os.path.join(output_dir, f"{protein}.pt")
            torch.save(embedding, output_path)
            print(f"Saved embedding for {protein}")

# encode_ontology_batch()
# exit()


'''
Label Encoding
'''
def encode_labels():

    model = get_model("llama2")

    gen = os.listdir("/home/fbqc9/Workspace/MCLLM_DATA/data_old/go_dataset")
    gen = set([i.split(".")[0] for i in gen])

    ontologies = ["CC", "MF", "BP"]

    go_graph = load_graph()
    node_desc = dict(go_graph.nodes(data="def"))

    for ontology in ontologies:
        embeddings = []
        terms = load_pickle(f"/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{ontology}_terms")

        for i, term in enumerate(terms):
            term_def = node_desc[term].split('"')[1]
            term_embedd = get_embeddings(model, term_def)
            embeddings.append(term_embedd)

        stacked_tensor = torch.stack(embeddings).squeeze(1)
        save_pickle(stacked_tensor, f"/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/{ontology}_embeddings")
# encode_labels()

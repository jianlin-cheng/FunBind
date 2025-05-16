import argparse
import torch
from transformers import EsmTokenizer, T5Tokenizer, AutoTokenizer
from transformers import EsmModel, T5EncoderModel, AutoModel
import re
import torch.nn.functional as F
from models.Metrics import Retrieve_MRR, Retrive_at_k
from data_processing.extract_data import combine_modalities
from models.model import SeqBindPretrain
from utils import load_ckp, load_config, load_graph
from Const import BASE_DATA_DIR


# TODO: 
# 1. Add validation to ensure topk <= len(terms)
# 2. Add validation to ensure the input data is in the correct format
# 3. Add inference with other base models



def load_model(device, ckp_file):
    config = load_config('config.yaml')['config1']
    model = SeqBindPretrain(config=config).to(device)
    # ckp_dir = '/home/fbqc9/Workspace/MCLLM_DATA/DATA/saved_models/'
    # ckp_file = ckp_dir + "pretrained_ontology.pt"
    print("Loading model checkpoint @ {}".format(ckp_file))
    load_model = load_ckp(filename=ckp_file, model=model, model_only=True)
    return load_model


def load_ontology_list(ontology_file):
    ontology = []
    with open(ontology_file, 'r') as f:
        for line in f:
            line = tuple(line.strip().split(" "))
            ontology.append(line)
    return ontology, []


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


def get_model(model_name, device):
    model_map = {
        "esm2_t48": ('facebook/esm2_t48_15B_UR50D', EsmTokenizer, EsmModel),
        "prost5":   ('Rostlab/ProstT5', T5Tokenizer, T5EncoderModel),
        "llama2":   ('meta-llama/Llama-2-7b-hf', AutoTokenizer, AutoModel),
    }

    model_info = model_map.get(model_name)
    if model_info is None:
        raise ValueError(f"Model {model_name} is not recognized.")
    
    model_path, tokenizer_class, model_class = model_info
    # print(f">>>>>>>>>>>>>>>>>........... The model path: {model_path}")

    if 'prost5' in model_name:
        tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=False)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_path)

    model = model_class.from_pretrained(model_path)
    
    if "llama" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model.tokenizer = tokenizer
    model.eval()
    return model.to(device)


def get_embeddings(model, data, modality, device):

    if modality == "Sequence":
        data = str(data)
    if modality == "Structure":
        data = str(data).lower()
        data = re.sub(r"[UZOB]", "X", data)
        data = " ".join(list(data))
        data = "<fold2AA>" + " " + data

    if modality == "Text" or modality == "Interpro" or modality == "Ontology":
        inputs = model.tokenizer(data, return_tensors="pt", max_length=1024, padding='max_length', truncation=True)
    elif modality == "Structure":
        inputs = model.tokenizer(data, return_tensors="pt", add_special_tokens=True, padding='longest')
    else:
        inputs = model.tokenizer(data, return_tensors="pt")

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    outputs = torch.mean(outputs.last_hidden_state, dim=1)
    return outputs


def generate_embeddings(data, modality="Sequence"):
    embeddings = {}

    if modality == "Sequence":
        model = get_model("esm2_t48", args.device)
    elif modality == "Structure":
        model = get_model("prost5", args.device)
    else:
        model = get_model("llama2", args.device)

    for protein, values in data.items():
        raw_data = values[modality]
        embeddings[protein] = get_embeddings(model, raw_data, modality, device=args.device)
    return embeddings


def compute_similarity(modality, model, modality_embeddings, ontology_embeddings, terms, term_names, topk=3):

    proteins, embeddings = list(modality_embeddings.keys()), list(modality_embeddings.values())

    embeddings = torch.stack(embeddings, dim=0)

    with torch.no_grad():
        mod1_features, _ = model.encode_modality(modality=f'{modality}_modality', value=embeddings.squeeze(1))
        ont_features, _ = model.encode_modality(modality='Ontology_modality', value=ontology_embeddings)

    mod1_features = F.normalize(mod1_features, dim=-1)
    ont_features = F.normalize(ont_features, dim=-1)

    similarity_matrix = (50.0 * mod1_features @ ont_features.T).softmax(dim=-1)

    topk_values, topk_indices = torch.topk(similarity_matrix, k=topk, dim=1)


    for i, protein in enumerate(proteins):
        print("Predictions for protein:", protein)
        for j in range(topk):
            print(f"Top {j+1} term: {terms[topk_indices[i][j]]}, Score: {topk_values[i][j].item() * 100:.2f}%")
        print("-----------------------------")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Zero-shot inference with FunBind.")

    parser.add_argument('--input-path', type=str, required=True, help="Path to the input file for the selected modality")
    parser.add_argument('--modality', type=str, required=True, choices=['Sequence', 'Structure', 'Text', 'Interpro'], help="Input modality type")
    parser.add_argument('--ontology-path', type=str, required=True, help="Path to list of ontology terms")
    parser.add_argument('--go-graph', type=str, required=True, help="Path to ontology file(OBO)")
    parser.add_argument('--model-checkpoint', type=str, required=True, help="Path to the pretrained model checkpoint.")
    parser.add_argument('--batch', type=int, help="BatchSize", default=3)
    parser.add_argument('--topk', type=int, help="Top K", default=1)
    parser.add_argument('--device', type=str, default='cpu', help="Device to run inference on (cuda or cpu).")

    args = parser.parse_args()


    if args.modality == 'Sequence':
        data = combine_modalities(sequence_data=args.input_path, use_sequence=False)
    elif args.modality == 'Structure':
        data = combine_modalities(structure_data=args.input_path, use_sequence=False)
    elif args.modality == 'Text':
        data = combine_modalities(textual_data=args.input_path, use_sequence=False)
    elif args.modality == 'Interpro':
        data = combine_modalities(interpro_data=args.input_path, use_sequence=False)

    modality_embeddings = generate_embeddings(data, args.modality)

    # Change to include term names
    terms, term_names = load_ontology_list(args.ontology_path)
    go_graph = load_graph(args.go_graph)
    ontology_text = process_ontology(go_graph, terms)
    ontology_embeddings = get_embeddings(get_model("llama2", args.device), ontology_text, "Ontology", device=args.device)

    # Load model
    model = load_model(args.device, args.model_checkpoint)
    model.eval()


    similarities = compute_similarity(modality=args.modality,
                                      model=model, 
                                      modality_embeddings=modality_embeddings, 
                                      ontology_embeddings=ontology_embeddings,
                                      terms=terms,
                                      term_names=terms, topk=args.topk)
    
    
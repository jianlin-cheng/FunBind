import re
import numpy as np
import torch
from transformers import EsmTokenizer, T5Tokenizer, AutoTokenizer
from transformers import EsmModel, T5EncoderModel, AutoModel
from models.model import SeqBindPretrain
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from utils import load_ckp, load_config, load_graph



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


def load_embedding(prot_name, modality1):
    x = torch.load(f'/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/dataset_new/{prot_name}.pt')

    #model_map = {"Sequence": "esm2_t48", "Structure": "prost5", "Text": "llama2", "Interpro": "llama2"}

    if modality1 == 'Sequence':
        return x[modality1]['esm2_t48']
    else:
        return x[modality1]


def compute_similarity(protein_list, term_list, modality1, model):

    go_graph = load_graph()

    modality1_embeddings = [load_embedding(prot, modality1) for prot in protein_list]
    modality1_embeddings = torch.stack(modality1_embeddings, dim=0).to(device)
    
    ontology_text = process_ontology(go_graph, term_list)
    ontology_embeddings = get_embeddings(get_model("llama2"), ontology_text, "Ontology")


    print(ontology_embeddings)

    exit()


    with torch.no_grad():
        mod1_features, _ = model.encode_modality(modality=f'{modality1}_modality', value=modality1_embeddings.squeeze(1))
        ont_features, _ = model.encode_modality(modality='Ontology_modality', value=ontology_embeddings)

    mod1_features = F.normalize(mod1_features, dim=-1)
    ont_features = F.normalize(ont_features, dim=-1)

    similarity = (50.0 * mod1_features @ ont_features.T).softmax(dim=-1)

    return similarity



def visualize_top_go_terms(protein_names, sims, term_list, term_names, go_term_colors, modality, top_k=3, y_min=0, y_max=120, figsize=(10, 6)):

    go_terms_dict = {}
    go_term_names = {}
    
    for i, (protein, sim) in enumerate(zip(protein_names, sims)):
        print(protein)
        values, indices = sim.topk(top_k)
        protein_terms = []
        
        for value, index in zip(values, indices):
            confidence = value.item() * 100
            ontology_term = ', '.join(term_list[index])

            go_term_names[ontology_term] = ", ".join(term_names[i] for i in term_list[index])


            if confidence > 1:
                protein_terms.append((confidence, ontology_term))

                print(f"- {ontology_term} ({ontology_term}): {confidence:.2f}%")
            
        go_terms_dict[f'{protein}'] = protein_terms

        print("\n")
    

    all_go_terms = set()
    for terms in go_terms_dict.values():
        for _, go_id in terms:
            all_go_terms.add(go_id)
    
  
    fig, ax = plt.subplots(figsize=figsize)
    group_width = 0.95
    

    for i, protein in enumerate(protein_names):

        bar_width = group_width / len(go_terms_dict[protein])

        protein_key = f'{protein}'
        protein_terms = go_terms_dict[protein_key]
        num_terms = len(protein_terms)

        term_positions = np.arange(num_terms) * bar_width - (num_terms * bar_width) / 2 + bar_width / 2

        for j, (confidence, go_id) in enumerate(protein_terms):
            x_pos = i + term_positions[j]
            bar = ax.bar(x_pos, confidence, bar_width * 0.98, color=go_term_colors[go_id], 
                         label=f"{go_id}" if go_id not in [l.get_label() for l in ax.get_lines() + ax.patches] else "")
            
            label_y_pos = min(confidence + 0.2, y_max - 5)
    
            ax.annotate(f"{go_id}",
                    xy=(x_pos, label_y_pos),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    color='black',
                    rotation=90,
                    fontsize=16,
                    #fontweight='bold'
                    )
    

    ax.set_ylabel('Confidence (%)', fontsize=16)
    ax.set_xticks(range(len(protein_names)))
    ax.set_xticklabels(protein_names, rotation=0, ha='center', fontsize=16)

    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_xlim(left=-0.48, right=len(protein_names) - 1 + bar_width + 0.25)
    handles, labels = [], []
    for go_id in sorted(all_go_terms):
        patch = plt.Rectangle((0, 0), 1, 1, fc=go_term_colors[go_id])
        handles.append(patch)
        labels.append(go_id)
    ax.legend(handles, labels, title="GO Terms", loc='best')
    
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(f"go_terms_{modality}.png", dpi=300)
    plt.clf()
    plt.close(fig)


    


def visualize_go_legends(go_term_names, go_term_colors):
    
    sorted_items = sorted(go_term_names.items(), key=lambda x: len(x[1]))

    handles = []
    for key, name in sorted_items:
        label = f"{key}: {name}"
        color = go_term_colors.get(key, "gray")
        handles.append(plt.Line2D([0], [0], color=color, lw=4, label=label))

    plt.figure(figsize=(5, 2))
    plt.legend(handles=handles, loc="center", ncol=1, fontsize=24, frameon=False, handlelength=4)

    plt.axis("off")
    plt.savefig("go_legend.png", dpi=300, bbox_inches="tight")
    plt.close()



def main():
    
    modality = "Interpro"

    protein_list = ["A8BPK8",  "P18335", "Q12198", "Q64565"]
    term_list = [('GO:0097558',), ('GO:0097559',), ('GO:0097560',),
                 ('GO:0097561',), ('GO:0170033',), ('GO:0170035',), 
                 ('GO:0170038',), ('GO:0170039',), ('GO:0170041',),
                 ('GO:0170043',), ('GO:1905504',), ('GO:1902674',)]
    

    term_names = {
        'GO:0097558': 'left ventral flagellum',
        'GO:0097559': 'right ventral flagellum',
        'GO:0097560': 'left caudal flagellum',
        'GO:0097561': 'right caudal flagellum',
        'GO:0170033': 'L-amino acid metabolic process',
        'GO:0170035': 'L-amino acid catabolic process',
        'GO:0170038': 'proteinogenic amino acid biosynthetic process',
        'GO:0170039': 'proteinogenic amino acid metabolic process',
        'GO:0170041': 'non-proteinogenic amino acid metabolic process',
        'GO:0170043': 'non-proteinogenic amino acid biosynthetic process',
        'GO:1905504': 'negative regulation of motile cilium assembly',
        'GO:1902674': 'right posteriolateral basal body',
    }


    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", 
                "#ff9896", "#98df8a", "#c5b0d5", "#ffbb78", "#f7b6d2", 
                "#dbdb8d", "#9edae5", "#c49c94", "#aec7e8", "#ffae19"
                ]  
             
    go_term_colors = {go_id: colors[i % len(colors)] 
                     for i, go_id in enumerate(term_names.keys())}
    

    model = load_model()
    model.eval()

    sims = compute_similarity(protein_list=protein_list, term_list=term_list, modality1=modality, model=model)
    visualize_top_go_terms(protein_list, sims, term_list, term_names, go_term_colors=go_term_colors, modality=modality, top_k=5)
    visualize_go_legends(term_names, go_term_colors=go_term_colors)

    
if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cuda:1"

    main()

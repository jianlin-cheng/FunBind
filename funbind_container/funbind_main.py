import argparse
import os
import pandas as pd

import re
import torch
from extract_data import combine_modalities
from utils import load_pickle
from model import SeqBindClassifier
from utils import load_ckp, load_config, load_graph
from transformers import EsmTokenizer, T5Tokenizer, AutoTokenizer
from transformers import EsmModel, T5EncoderModel, AutoModel, AutoTokenizer
from dataset import  CustomDataCollator
from torch.utils.data import Dataset, DataLoader
import networkx as nx


thresholds = {
         "BP": [0.2, 0.1, 0.6, 0.1],
         "CC": [0.3, 0.1, 0.5, 0.1],
         "MF": [0.2, 0.1, 0.4, 0.3]
    }

FUNC_DICT = {
    'CC': 'GO:0005575',
    'MF': 'GO:0003674',
    'BP': 'GO:0008150'
    }



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
    def __init__(self, modality,  data_path, data_list):

        self.modality = modality
        self.base_path = data_path
        self.data_list = data_list

        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        protein = self.data_list[idx]

        if self.modality == "Combined":
            seq_path = os.path.join(self.base_path, "Sequence", f"{protein}.pt")
            struct_path = os.path.join(self.base_path, "Structure", f"{protein}.pt")
            text_path = os.path.join(self.base_path, "Text", f"{protein}.pt")
            interpro_path = os.path.join(self.base_path, "Interpro", f"{protein}.pt")

            seq = torch.load(seq_path)
            struct = torch.load(struct_path)
            text = torch.load(text_path)
            interpro = torch.load(interpro_path)
            return (protein, seq, struct, text, interpro)
        else:
            pth = os.path.join(self.base_path, self.modality, f"{protein}.pt")
            sample = torch.load(pth, weights_only=False)
            return (protein, sample)
            

def _preprocess(data, modality):
    data = str(data)

    if modality == "Sequence":
        data = str(data).upper()
    elif modality == "Structure":
        data = str(data).lower()

    data = re.sub(r"[UZOB]", "X", data)
    data = " ".join(list(data))

    if modality == "Sequence":
        data = "<AA2fold>" + " " + data
    elif modality == "Structure":
        data = "<fold2AA>" + " " + data

    return data


def get_embeddings(model, base_model, data, modality, device='cpu'):

    if modality == "Sequence":
        if base_model == "prostt5":
            data = _preprocess(data, modality)
        else:
            data = str(data)
    if modality == "Structure":
        data = _preprocess(data, modality)

    if modality == "Text" or modality == "Interpro" or modality == "Ontology":
        inputs = model.config.tokenizer(data, return_tensors="pt", max_length=1024, padding='max_length', truncation=True)
    else:
        inputs = model.config.tokenizer(data, return_tensors="pt")

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    outputs = torch.mean(outputs.last_hidden_state, dim=1).cpu()

    return outputs


def get_model(model_name, device='cpu'):
    model_map = {
        "esm2_t48": ('facebook/esm2_t48_15B_UR50D', EsmTokenizer, EsmModel),
        "prostt5":   ('Rostlab/ProstT5', T5Tokenizer, T5EncoderModel),
        "llama2":   ('meta-llama/Llama-2-7b-hf', AutoTokenizer, AutoModel),
    }

    model_info = model_map.get(model_name)
    if model_info is None:
        raise ValueError(f"Model {model_name} is not recognized.")
    
    model_path, tokenizer_class, model_class = model_info

    # Set local cache path
    cache_dir = "/root/.cache/huggingface/hub"
    
    # 1️⃣ Try Hugging Face cached format first
    hf_model_path = os.path.join(cache_dir, f"models--{model_path.replace('/', '--')}")

    # for snapshot_path
    shot_path = os.path.join(cache_dir, model_name)

    # Load tokenizer offline
    if 'prostt5' in model_name:
        if os.path.exists(hf_model_path):
            tokenizer = tokenizer_class.from_pretrained(model_path, cache_dir=cache_dir, local_files_only=True, do_lower_case=False)
        else:
            tokenizer = tokenizer_class.from_pretrained(shot_path, local_files_only=True, do_lower_case=False)
    else:
        if os.path.exists(hf_model_path):
            tokenizer = tokenizer_class.from_pretrained(model_path, cache_dir=cache_dir, local_files_only=True)
        else:
            tokenizer = tokenizer_class.from_pretrained(shot_path, local_files_only=True)
    # Load model offline
    if os.path.exists(hf_model_path):
        model = model_class.from_pretrained(model_path, cache_dir=cache_dir, local_files_only=True)
    else:
        model = model_class.from_pretrained(shot_path, local_files_only=True)
    
    r'''
    # bellow load model and tokenization online
    if 'prostt5' in model_name:
        tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=False)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_path)
    
    model = model_class.from_pretrained(model_path)
    '''
    model.config.tokenizer = tokenizer
    
    if "llama" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model.tokenizer = tokenizer
    model.eval()
    return model.to(device)


def load_model(ontology, ckp_dir, base_model, device):
    config = load_config('config.yaml')
    pretrain_config = config['pretraining_configs'][base_model]
    classifier_config = config['classification_configs'][ontology]
    model = SeqBindClassifier(pretrain_config=pretrain_config, classifier_config=classifier_config).to(device)
    
    ckp_file = f'{ckp_dir}/{ontology}_{base_model}.pt'
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



def generate_embeddings(data, modality, base_model, directory, device='cpu'):

    encoder =  {'Sequence': 'esm2_t48', 'Structure': 'prostt5', 'Interpro': 'llama2', 'Text': 'llama2'}
    
    if modality ==  'Sequence' and base_model == 'prostt5':
        encoding_model = get_model("prostt5", device=device)
    else:
        encoding_model = get_model(encoder[modality], device=device)

    proteins = []

   
    for prot_name in data:
        # skip existing embeddings
        if os.path.exists(os.path.join(directory, f"{prot_name}.pt")):
                continue
        embedding = get_embeddings(encoding_model, base_model, data[prot_name][modality], modality, device=device)
        torch.save(embedding, os.path.join(directory, f"{prot_name}.pt"))
        proteins.append(prot_name)

    # check missing items to make sure we get all the protein items.
    allkey = list(data.keys())
    if len(proteins) < len(allkey):
        protall = []
        protall.extend(allkey)
        return protall

    return proteins



def validate_args(args):
    if not (args.sequence_path or args.structure_path or args.text_path or args.interpro_path):
        raise ValueError("At least one of --sequence-path, --structure-path, --text-path, or --interpro-path must be provided.")

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")
    
    for path in [args.sequence_path, args.structure_path, args.text_path, args.interpro_path]:
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        


def prepare_embeddings(args):
    modalities_proteins = {'Sequence': [], 'Structure': [], 'Text': [], 'Interpro': []}
    modalities = []

    input_map = {
        "Sequence": args.sequence_path,
        "Structure": args.structure_path,
        "Text": args.text_path,
        "Interpro": args.interpro_path,
    }

    for modality, path in input_map.items():
        if path:
            directory_path = os.path.join(args.working_dir, modality)
            os.makedirs(directory_path, exist_ok=True)
            data = combine_modalities(**{modality.lower() + "_data": path}, use_sequence=False)
            modalities_proteins[modality] = generate_embeddings(data, modality, args.base_model, directory_path, device=args.device)
            modalities.append(modality)

    modalities_proteins['Combined'] = list(set(modalities_proteins['Sequence']) & set(modalities_proteins['Structure']) & set(modalities_proteins['Text']) & set(modalities_proteins['Interpro']))
    return modalities_proteins, modalities


def save_results(predictions_dict, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    for mod, protein_predictions in predictions_dict.items():
        with open(f"{args.output}/{mod}_{args.ontology}.tsv", "w") as f:
            for protein, go_term_scores in protein_predictions.items():
                for go_term, score in go_term_scores.items():
                    if score >= 0.01:
                        f.write(f"{protein}\t{go_term}\t{score:.4f}\n")


def perform_inference(args, modalities_proteins, modalities):

    model = load_model(ontology=args.ontology, ckp_dir=args.data_path, base_model=args.base_model, device=args.device)
    model.eval()

    predictions_dict = {mod: {} for mod in modalities}

    go_terms_list =  load_pickle(f"{args.data_path}/{args.ontology}_terms")
    go_graph = load_graph(f"{args.data_path}/go-basic.obo")
    go_set = nx.ancestors(go_graph, FUNC_DICT[args.ontology])


    with torch.no_grad():
        for mod in modalities:
            dataset = InferenceDataset(modality=mod, data_path=args.working_dir, 
                                        data_list=modalities_proteins[mod])
            
            collator = CustomDataCollator(modality=mod, device=args.device)
            dataloader = DataLoader(dataset, batch_size=args.num_batches, shuffle=False, collate_fn=collator)

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

    
    if len(modalities_proteins['Combined']) > 0:

        predictions_dict['Combined'] = fuse_predictions(predictions_dict, 
                                    modality_weights={"Sequence": thresholds[args.ontology][0], 
                                                    "Structure": thresholds[args.ontology][1],
                                                    "Text": thresholds[args.ontology][2], 
                                                    "Interpro": thresholds[args.ontology][3]},
                                    go_terms_list=go_terms_list,
                                    go_graph=go_graph,
                                    go_set=go_set)


    return predictions_dict

def merge3terms(path, modalities, top=1500):
    # create a directory to store final results
    merge_dir = f"{path}/merged3terms"
    if not os.path.exists(merge_dir):
        os.makedirs(merge_dir, exist_ok=True)

    # print(modalities)
    print(f"\n>>>>>>>> ✅Step III: Collecting prediction results for all three ontology terms.")
    for md in modalities:
        # === Step 1: Read the three TSV files ===
        df1 = pd.read_csv(f"{path}/BP/{md}_BP.tsv", sep="\t", names=["Protein ID", "GO_terms", "Scores"])
        df2 = pd.read_csv(f"{path}/CC/{md}_CC.tsv", sep="\t", names=["Protein ID", "GO_terms", "Scores"])
        df3 = pd.read_csv(f"{path}/MF/{md}_MF.tsv", sep="\t", names=["Protein ID", "GO_terms", "Scores"])

        # === Step 2: Combine all ===
        df = pd.concat([df1, df2, df3], ignore_index=True)

        # === Step 3: Sort by Protein ID and Scores (descending) ===
        df = df.sort_values(by=["Protein ID", "Scores"], ascending=[True, False])
    
        # === Step 4: Keep only top 1500 GO_terms per Protein ID ===
        df = df.groupby("Protein ID").head(top)

        # === Step 5: Round Scores to 3 decimal places ===
        df["Scores"] = df["Scores"].round(3)

        # === Step 6: Save the final file (no header, TSV format) ===
        df.to_csv(f"{merge_dir}/{md}.tsv.gz", sep="\t", index=False, header=False, compression="gzip")
        # df.to_csv(f"{merge_dir}/{md}.tsv", sep="\t", index=False, header=False)

        if md not in "Combined":
            print(f"======== ✅Saved {md} modality results for all three ontology terms at {merge_dir}/{md}.tsv.gz.")
        else:
            print(f"======== ✅Saved: {md} all-modality predictions for all three ontology terms at {merge_dir}/{md}.tsv.gz.")
    
    print(f"\n✅✅✅✅ All predictions completed! ✅✅✅✅")
    

def main():
    parser = argparse.ArgumentParser(description="Supervised Classification with FunBind.")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the pretrained model checkpoint & Go term list & other relevant data.")
    parser.add_argument('--sequence-path', type=str, default=None, help="Path to the input file(Sequence)")
    parser.add_argument('--structure-path', type=str, default=None, help="Path to the input file(Structure)")
    parser.add_argument('--text-path', type=str, default=None, help="Path to the input file(Text)")
    parser.add_argument('--interpro-path', type=str, default=None, help="Path to the input file(Interpro)")
    parser.add_argument('--ontology', type=str, default="CC", help="Path to data files")
    parser.add_argument('--base-model', type=str, choices=['esm2', 'prostt5'], default='prostt5', help='Base model to use.')
    parser.add_argument('--num-batches', type=int, default=32, help="Number of batches for inference")
    parser.add_argument('--device', type=str, default='cpu', help="Device to run inference on (cuda or cpu).")
    parser.add_argument('--working-dir', type=str, default="./", help="Path to generate temporary files")
    parser.add_argument('--output', type=str, default="results", help="File to save output")
    # parser.add_argument('--token', type=str, default=None, help="token for your login huggingface account to use text or interpro modality")

    args = parser.parse_args()
    args.working_dir = "embeddings" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device


    # args.data_path = "/root/.cache/checkpoints"
    # cache_path = "/root/.cache/checkpoints"
    # args.sequence_path = f"{cache_path}/examples/classification/sequence.fasta"
    # args.interpro_path = f"{cache_path}/examples/classification/interpro.txt"
    # args.text_path = f"{cache_path}/examples/classification/text.txt"
    # args.structure_path = f"{cache_path}/examples/classification/structure.fasta"
   
    # cmd for prediction from text: 
    # python funbind_main.py --data-path /root/.cache/checkpoints --text-path /root/.cache/checkpoints/examples/classification/text.txt
    # python funbind_main.py --data-path /root/.cache/checkpoints --interpro-path /root/.cache/checkpoints/examples/classification/interpro.txt
    # python funbind_main.py --data-path /root/.cache/checkpoints --structure-path /root/.cache/checkpoints/examples/classification/structure.fasta
    # python funbind_main.py --data-path /root/.cache/checkpoints --sequence-path /root/.cache/checkpoints/examples/classification/sequence.fasta

    validate_args(args)
    modalities_proteins, modalities = prepare_embeddings(args)
    print(f"\n>>>>>>>> ✅Step I: Finished extracting feature embeddings from the pretrained models! Results are saved in: {args.working_dir}.\n")
    
    ontology = ["CC", "BP", "MF"]
    parent_dir = args.output

    print(f">>>>>>>> ✅Step II: Running ontology-specific function prediction using FunBind.")
    for nt in ontology:
        args.ontology = nt
        args.output = parent_dir + f"/{nt}"
        predictions_dict = perform_inference(args, modalities_proteins, modalities)
        save_results(predictions_dict, args.output, args)
        print(f"******** ✅Completed specific {nt} ontology prediction with FunBind. Both individual modality results and the combined all-modality predictions are saved to {args.output}.\n")
    
    if len(modalities) > 1:
        modalities.append('Combined')
    merge3terms(parent_dir, modalities, top = 1500)
    modalities.pop()

   
if __name__ == "__main__":
    main()
    


    

    


    
        

    
                        








    

    

    
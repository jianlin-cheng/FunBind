import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import EsmModel, EsmTokenizer, EsmConfig
from transformers import BertTokenizer, BertModel
from transformers import DataCollatorForLanguageModeling
import clip

class CustomDataset(Dataset):
    def __init__(self, args):

        self.modal_pair = args.modality_pair
        self.data_pair = get_samples(args)
        self.device = "cuda:0"

        self.esm_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = None  # Assuming no transformation for now

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        if self.modal_pair == "ss":
            sequence_modality, matched_modality = self.data_pair[idx]
            sequence_modality = self.esm_tokenizer(sequence_modality, return_tensors='pt', padding="max_length", truncation=True, max_length=128, return_special_tokens_mask=True)
            structure_modality = self.esm_tokenizer(matched_modality, return_tensors='pt', padding="max_length", truncation=True, max_length=128, return_special_tokens_mask=True)
            
            sequence_modality = {'input_ids': sequence_modality['input_ids'].squeeze().to(self.device),  'attention_mask': sequence_modality['attention_mask'].squeeze().to(self.device)}
            structure_modality = {'input_ids': structure_modality['input_ids'].squeeze().to(self.device),  'attention_mask': structure_modality['attention_mask'].squeeze().to(self.device)}
            
            return sequence_modality, structure_modality
        
        elif self.modal_pair == "st":
            sequence_modality, matched_modality = self.data_pair[idx]
            sequence_modality = self.esm_tokenizer(sequence_modality, return_tensors='pt', padding="max_length", truncation=True, max_length=128, return_special_tokens_mask=True)
            #text_modality = self.bert_tokenizer(matched_modality, return_tensors='pt', padding="max_length", truncation=True, max_length=128, return_special_tokens_mask=True)
            text_modality = clip.tokenize(matched_modality)
            
            sequence_modality = {'input_ids': sequence_modality['input_ids'].squeeze().to(self.device),  'attention_mask': sequence_modality['attention_mask'].squeeze().to(self.device)}
            #text_modality = {'input_ids': text_modality['input_ids'].squeeze().to(self.device),  'attention_mask': text_modality['attention_mask'].squeeze().to(self.device)}
            text_modality = {'input_ids': text_modality.squeeze().to(self.device)}


            return sequence_modality, text_modality



class CustomDataCollator:
    def __init__(self, matching_modality="ss"):
        
        self.esm_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.sequence_data_collator = DataCollatorForLanguageModeling(tokenizer=self.esm_tokenizer, mlm_probability=0.15)
        self.structure_data_collator = DataCollatorForLanguageModeling(tokenizer=self.esm_tokenizer, mlm_probability=0.15)
        self.text_data_collator = DataCollatorForLanguageModeling(tokenizer=self.bert_tokenizer, mlm_probability=0.15)
        
        self.matching_modality = matching_modality
        self.device = "cuda:0"



    def __call__(self, batch):


        input_ids_1, input_ids_2 = [], []
        for item in batch:
            input_ids_1.append(item[0])
            input_ids_2.append(item[1])

        if self.matching_modality == "ss":
            sequence_modality = self.sequence_data_collator(input_ids_1)
            structure_modality = self.structure_data_collator(input_ids_2)

            return {
                'sequence_modality': {k: v.to(self.device) for k, v in sequence_modality.items()},
                'structure_modality': {k: v.to(self.device) for k, v in structure_modality.items()}
            }
        elif self.matching_modality == "st":
            sequence_modality = self.sequence_data_collator(input_ids_1)
            text_modality = self.text_data_collator(input_ids_2)
            return {
                'sequence_modality': {k: v.to(self.device) for k, v in sequence_modality.items()},
                'text_modality': {k: v.to(self.device) for k, v in text_modality.items()}
            }



import random
import string

def generate_protein_sequence(min_length=500, max_length=1024):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(amino_acids, k=length))

# Function to generate a random structure representation
def generate_structure_representation(min_length=500, max_length=1024):
    structure_chars = "ACDEFGHIKLMNPQRSTVWY"
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(structure_chars, k=length))

# Function to generate a random descriptive text
def generate_descriptive_text(min_length=30, max_length=300):
    length = random.randint(min_length, max_length)
    words = ["protein", "sequence", "structure", "ID", "function", "binding", "site", "domain", "motif", "enzyme"]
    text = " ".join(random.choices(words, k=length // 5))  # Approximate word count
    return text + " " + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))



def get_samples(args):
    if args.modality_pair == "ss":
        # Generate 100 samples for each modality
        num_samples = 500000
        sequences = [generate_protein_sequence() for _ in range(num_samples)]
        structures = [generate_structure_representation() for _ in range(num_samples)]
        samples = list(zip(sequences, structures))
    
    elif args.modality_pair == "st":
        # Generate 100 samples for each modality
        num_samples = 500000
        sequences = [generate_protein_sequence() for _ in range(num_samples)]
        texts = [generate_descriptive_text() for _ in range(num_samples)]
        samples = list(zip(sequences, texts))

    return samples
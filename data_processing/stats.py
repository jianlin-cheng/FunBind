from collections import defaultdict
import re
import networkx as nx
import numpy as np
import obonet
from Bio import SwissProt
from sklearn.model_selection import train_test_split
from utils import save_pickle, load_pickle
from Bio.UniProt import GOA
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import subprocess



def data_stats():

    combined_data = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/raw/combined_data")
    text = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/raw/text_data")
    cc = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/CC_groundtruth")
    mf = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/MF_groundtruth")
    bp = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/labels/BP_groundtruth")

    df_comb = pd.DataFrame.from_dict(combined_data, orient='index')
    df_text = pd.DataFrame.from_dict(text, orient='index')


    df = pd.merge(df_comb, df_text, how='left', left_index=True, right_index=True)

    df["CC"] = np.where(df.index.isin(cc.keys()), "CC", None)
    df["MF"] = np.where(df.index.isin(mf.keys()), "MF", None)
    df["BP"] = np.where(df.index.isin(bp.keys()), "BP", None)

    stats = df.describe(include='all').loc['count']

    print(stats)

    data_list = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/train_valid")

    for key, value in data_list.items():
        print(key, len(value))


    print("##########--CC--##########\n")
    print(df[df['CC'].notnull()].describe(include='all').loc['count'])
    print(df[df['CC'].notnull() & df['Sequence'].notnull() & df['Structure'].notnull() & df['Text'].notnull() & df['Interpro'].notnull()].describe(include='all').loc['count'])
    print("##########--CC--##########\n")

    print("##########--MF--##########\n")
    print(df[df['MF'].notnull()].describe(include='all').loc['count'])
    print(df[df['MF'].notnull() & df['Sequence'].notnull() & df['Structure'].notnull() & df['Text'].notnull() & df['Interpro'].notnull()].describe(include='all').loc['count'])
    print("##########--MF--##########\n")

    print("##########--BP--##########\n")
    print(df[df['BP'].notnull()].describe(include='all').loc['count'])
    print(df[df['BP'].notnull() & df['Sequence'].notnull() & df['Structure'].notnull() & df['Text'].notnull() & df['Interpro'].notnull()].describe(include='all').loc['count'])
    print("##########--BP--##########\n")


# data_stats()


def count_tokens(data_pth):
    model_name = 'meta-llama/Llama-2-7b-hf' #"meta-llama/Meta-Llama-3-8B"

    data = load_pickle(data_pth)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ct =0
    for i  in data:

        tokens = tokenizer(data[i])

        token_ids = tokens['input_ids']
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)

        # Get the distinct tokens
        distinct_tokens = set(token_strings)

        if len(tokens['input_ids']) > 1000:
            ct+=1
        # Print the number of tokens
            print(f"Number of tokens: {len(tokens['input_ids'])}")
    print(ct)
# count_tokens("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/ontology_list_BP")
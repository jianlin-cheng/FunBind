from collections import defaultdict
import os
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


''''
Scripts used in processing raw data
'''
def remove_publication_records(record):
    pattern = r'\{ECO:[^}]*\}|\(PubMed:\d+\)'
    cleaned_description = {}
    for key, value in record.items():
        cleaned_value = re.sub(pattern, '', value).strip()
        cleaned_value = re.sub(r'\s*\.\s*\.', '.', cleaned_value).strip()
        cleaned_description[key] = cleaned_value
    return cleaned_description


def create_interpro_graph(file_name):
    def read_file(file_name):
        rels = open(file_name, 'r')
        lines = [i.rstrip('\n').split("::") for i in rels.readlines()]
        return lines
    
    data = read_file(file_name)

    graph = nx.DiGraph()

    for node in data:
        # 8 dashes
        if node[0].startswith("--------"):
            l5 = node[0].strip("--------")
            graph.add_edges_from([(l4, l5)])
        # 6 dashes
        elif node[0].startswith("------"):
            l4 = node[0].strip("------")
            graph.add_edges_from([(l3, l4)])
        # 4 dashes
        elif node[0].startswith("----"):
            l3 = node[0].strip("----")
            graph.add_edges_from([(l2, l3)])
        # 2 dashes
        elif node[0].startswith("--"):
            l2 = node[0].strip("--")
            graph.add_edges_from([(l1, l2)])
        else:
            l1 = node[0]
            if not graph.has_node(l1):
                graph.add_node(l1)
    return graph 


def get_interpro_desc():
    file_path = "/home/fbqc9/Workspace/MCLLM_DATA/INTERPRO/entry.list"

    data = {}
    with open(file_path, 'r') as file:

        next(file)
        for line in file:
            parts = line.strip().split('\t')
            data[parts[0]] = (parts[1], parts[2])
        return data


# Generate sequence data from Alphafold Structure.
def generate():
    CMD = "pwd"
    subprocess.call(CMD, shell=True, cwd="./")

    '''CMD = "foldseek createdb Swissprot queryDB"
    subprocess.call(CMD, shell=True, cwd="./")

    CMD = "foldseek lndb queryDB_h queryDB_ss_h"
    subprocess.call(CMD, shell=True, cwd="./")'''

    CMD = "foldseek convert2fasta queryDB_ss queryDB_ss.fasta"
    subprocess.call(CMD, shell=True, cwd="./")

# generate()

'''
Collect Textual Description
'''
def collect_textual_annotation(proteins, text_file):
    
    data = {}
    
    handle = open(text_file)
    for record in SwissProt.parse(handle):

        acc = record.accessions[0]

        if acc in proteins:
            annotations = {}
            annotations["PROTEIN NAME"] = record.entry_name
            annotations["DESCRIPTION"] = record.description.split("Full=")[1].split(";")[0]
            
            for comment in record.comments:
                if comment.startswith("FUNCTION:"):
                    annotations["FUNCTION"] = comment[len("FUNCTION:"):].strip()
                if comment.startswith("SUBUNIT:"):
                    annotations["SUBUNIT"] = comment[len("SUBUNIT:"):].strip()
                if comment.startswith("SUBCELLULAR LOCATION:"):
                    annotations["SUBCELLULAR LOCATION"] = comment[len("SUBCELLULAR LOCATION:"):].strip()
                if comment.startswith("TISSUE SPECIFICITY:"):
                    annotations["TISSUE SPECIFICITY"] = comment[len("TISSUE SPECIFICITY:"):].strip()
                if comment.startswith("INDUCTION:"):
                    annotations["INDUCTION"] = comment[len("INDUCTION:"):].strip()
                if comment.startswith("SIMILARITY:"):
                    annotations["SIMILARITY"] = comment[len("SIMILARITY:"):].strip()
            data[acc] = remove_publication_records(annotations)

    return data


'''
Collect Interpro Annotations
'''
def parse_protein2ipr(proteins, interpro_file, text_data):

    interpro_graph = create_interpro_graph(file_name="/home/fbqc9/Workspace/MCLLM_DATA/INTERPRO/ParentChildTreeFile.txt")
    interpro_desc = get_interpro_desc()


    protein_to_ipr = {}
    data = {}
    
    with open(interpro_file, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')

            protein_id = parts[0]

            if protein_id in proteins:
                interpro_id = parts[1]

                try:
                    interpro_ancestors = nx.ancestors(interpro_graph, interpro_id).union({interpro_id})
                except nx.exception.NetworkXError:
                    interpro_ancestors = set([interpro_id])

                if protein_id in protein_to_ipr:
                    protein_to_ipr[protein_id].update(interpro_ancestors)
                else:
                    protein_to_ipr[protein_id] = interpro_ancestors

    for protein_id, interpro_infos in protein_to_ipr.items():
        try:
            _prot = text_data[protein_id]
            concatenated_text = f"Protein Name: {_prot['PROTEIN NAME']}; Description: {_prot['DESCRIPTION']}" 
        except KeyError:
            concatenated_text = ""
        for interpro_info in interpro_infos:
            _temp = interpro_desc[interpro_info]
            temp_text = f"; Interpro Name: {_temp[1]}, Interpro Type: {_temp[0]}"
            concatenated_text += temp_text

        data[protein_id] = concatenated_text

    return data

def fasta_to_dic(fasta_file, sep="|", pos=0):
    data = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        data[seq_record.id.split(sep)[pos]] =  seq_record.seq
    return data


def extract_proteins_from_fasta(file_path):
    protein_sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        protein_sequences.append(str(record.id.split("|")[1]))
    return protein_sequences



def combine_modalities(sequence_data, structure_data, textual_data, interpro_data):
    """
    Combine data from multiple modalities into a unified representation.

    Args:
        sequence_data (str): Path to fasta sequence file
        structure_data (str): Path to Structural sequence file
        textual_data (): Path to textual data
        interpro_data (): Path to interpro data

    Returns:
        : A combined dictionary where keys are identifiers and values are unified
              representations incorporating all provided modalities.
    """


    proteins = set(extract_proteins_from_fasta(sequence_data))
    sequences = fasta_to_dic(sequence_data, sep="|", pos=1)
    structures = fasta_to_dic(structure_data, sep="-", pos=1)
    text = collect_textual_annotation(proteins, textual_data)
    interpros = parse_protein2ipr(proteins, interpro_data, text)


    data = {}
    for protein in proteins:
        
        sequence = sequences.get(protein, None)
        structure = structures.get(protein, None)
        structure = structure.lower() if structure  else None
        interpro =  interpros.get(protein, None)
        tmp = text.get(protein, None)

        textual = None
        if tmp:
            textual_fields = []
            for key in ['FUNCTION', 'SUBUNIT', 'SUBCELLULAR LOCATION', 'INDUCTION', 'TISSUE SPECIFICITY', 'SIMILARITY']:
                if key in tmp:
                    textual_fields.append(f"{key}: {tmp[key]}")

            textual = '; '.join(textual_fields) if textual_fields else None

            if textual:
                textual = f"PROTEIN NAME: {tmp['PROTEIN NAME']}; DESCRIPTION: {tmp['DESCRIPTION']}; " + textual


        tmp = {
            "Sequence": sequence,
            "Structure": structure,
            "Text": textual,
            "Interpro": interpro
        }
        data[protein] = {key: value for key, value in tmp.items() if value is not None}

        #data[protein] = {"Sequence": sequence, "Structure": structure, 
         #                "Text": textual, "Interpro": interpro}

    return data





'''
if __name__ == "__main__":

    sequence_fasta = "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test_data/sequence.fasta"
    structure_fasta = "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test_data/structure.fasta"
    text_data = "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test_data/text.dat"
    interpro_data = "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test_data/interpro.dat"

    data = combine_modalities(sequence_data=sequence_fasta, structure_data=structure_fasta,
                       textual_data=text_data, interpro_data=interpro_data)
    

    for i in data:
        print(i, data[i])
        exit()

'''


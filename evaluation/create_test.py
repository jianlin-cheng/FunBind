import os
import pickle
from Bio.UniProt import GOA
import obonet
import torch
import networkx as nx
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def save_pickle(obj, filepath):
    try:
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object successfully saved to {filepath}.")
    except Exception as e:
        print(f"Error saving object: {e}")
        

def load_pickle(filepath):
    try:
        with open(filepath, 'rb') as file:
            obj = pickle.load(file)
        print(f"Object successfully loaded from {filepath}.")
        return obj
    except Exception as e:
        print(f"Error loading object: {e}")
        return None
    

'''
function : given a file handle, parse in using gaf format and return a dictionary
           that identify those protein with experimental evidence and the ontology
input    : file text
output   : dic (key: name of file (number), value is a big dictionary store info about the protein)
'''
def read_gaf(handle):
    dic = {}
    all_protein_name = set()
    Evidence = {'Evidence': set(["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC", "HTP", "HDA", "HMP", "HGI", "HEP"])}
    with open(handle, 'r') as handle:
        for rec in GOA.gafiterator(handle):
            if GOA.record_has(rec, Evidence) and rec['DB'] == 'UniProtKB':
                all_protein_name.add(rec['DB_Object_ID'])
                if rec['DB_Object_ID'] not in dic:
                    dic[rec['DB_Object_ID']] = {rec['Aspect']: set([rec['GO_ID']])}
                else:
                    if rec['Aspect'] not in dic[rec['DB_Object_ID']]:
                        dic[rec['DB_Object_ID']][rec['Aspect']] = set([rec['GO_ID']])
                    else:
                        dic[rec['DB_Object_ID']][rec['Aspect']].add(rec['GO_ID'])
    return dic, all_protein_name


def read_gpi(in_file, proteins):
    swissprot = set()
    with open(in_file, 'r') as handle:
        for entry in GOA.gpi_iterator(handle):
            if entry['DB'] == 'UniProtKB' and entry['DB_Object_ID'] in proteins and entry['Gene_Product_Properties'][0] == "db_subset=Swiss-Prot":
                swissprot.add(entry['DB_Object_ID'])
    return swissprot


def analyze(t1_dic, t2_dic, swissprot_proteins):
    data = {
        "P": {},
        "C": {},
        "F": {} 
    }

    for protein in t2_dic:
        if protein in swissprot_proteins:
            if protein not in t1_dic:
                for ontology in t2_dic[protein]:
                    data[ontology][protein] = t2_dic[protein][ontology]
                    
            else:
                if len(t1_dic[protein]) < 3:
                    for ontology in t2_dic[protein]:
                        if ontology not in t1_dic[protein]:
                            data[ontology][protein] = t2_dic[protein][ontology]
    
    key_mapping = {"P": "BP", "C": "CC", "F": "MF"}
    data = {key_mapping.get(k, k): v for k, v in data.items()}

    return data


def load_graph():
    # go_graph = obonet.read_obo(open("/home/fbqc9/Workspace/MCLLM_DATA/DATA/cafa5/Train/go-basic.obo", 'r'))

    go_graph = obonet.read_obo(open("/home/fbqc9/Workspace/MCLLM/evaluation/go-basic.obo", 'r'))
    accepted_edges = set()
    unaccepted_edges = set()

    for edge in go_graph.edges:
        if edge[2] == 'is_a' or edge[2] == 'part_of':
            accepted_edges.add(edge)
        else:
            unaccepted_edges.add(edge)
    go_graph.remove_edges_from(unaccepted_edges)
    return go_graph




def write_annotations(data_dic, pth):
    go_graph = load_graph()

    ontologies = ['MF', 'CC', 'BP']

    test_proteins = {i: set() for i in ontologies}
    groundtruth = {i: {} for i in ontologies}

    

    for ont in data_dic:
        for acc, terms in data_dic[ont].items():
            
            for term in terms:
                try:
                    tmp = nx.descendants(go_graph, term).union({term})
                    if acc in groundtruth[ont]:
                        groundtruth[ont][acc].update(tmp)
                    else:
                        groundtruth[ont][acc] = tmp
                    test_proteins[ont].add(acc)
                except nx.NetworkXError:
                    print(ont, acc, term) 



    save_pickle(test_proteins, "/home/fbqc9/Workspace/MCLLM/evaluation/proteins")
    

    for ont in ontologies:
        file_name = os.path.join(pth, f"{ont}.tsv")
        with open(file_name, 'w') as file_out:
            for prot in test_proteins[ont]:
                for annot in groundtruth[ont][prot]:
                    file_out.write(f"{prot}\t{annot}\n")

    


'''
t1_dic, all_protein_t1 = read_gaf("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/goa_uniprot_all.gaf.212")
save_pickle(t1_dic, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/annotation_212")
save_pickle(all_protein_t1, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/proteins_212")


t2_dic, all_protein_t2 = read_gaf("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/goa_uniprot_all.gaf")
save_pickle(t2_dic, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/annotation_224")
save_pickle(all_protein_t2, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/proteins_224")
'''



t1_dic = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/annotation_212")
all_protein_t1 = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/proteins_212")
print(len(t1_dic), len(all_protein_t1))

print(t1_dic['Q9D9I1'])

t2_dic = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/annotation_224")
all_protein_t2 = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/proteins_224")
print(len(t2_dic), len(all_protein_t2))
print(t2_dic['Q9D9I1'])

# swissprot = read_gpi("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/goa_uniprot_all.gpi", all_protein_t2)
# save_pickle(swissprot, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/swissprot_224")
swissprot = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/swissprot_224")


data = analyze(t1_dic, t2_dic, swissprot)
# save_pickle(data, "/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/test_data")
# data = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/test/test_data")


for i in data:
    print(i, len(data[i]))
    for j in data[i]:
        if i == "CC" and j == "Q9D9I1":
            print(i, j, data[i][j])



# write_annotations(data, "/home/fbqc9/Workspace/MCLLM/evaluation/groundtruth")
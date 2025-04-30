import os
import pickle
import shutil
import subprocess
import sys
import networkx as nx
from Bio import SeqIO

# BASE_PATH
sys.path.append(os.path.abspath('/home/fbqc9/Workspace/MCLLM/'))

from Const import BASE_DATA_DIR


def pickle_save(data, filename):
    with open('{}'.format(filename), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def count_fasta_sequences(fasta_file):
    count = sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))
    return count

def get_all_sequences(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.id))
    return sequences


def filter_fasta(proteins, infile, outfile):
    seqs = []
    input_seq_iterator = SeqIO.parse(infile, "fasta")

    for pos, record in enumerate(input_seq_iterator):
        if record.id in proteins:
            seqs.append(record)

    print("Number of sequences in filtered fasta: {}".format(len(seqs)))
    SeqIO.write(seqs, outfile, "fasta")


def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def extract_from_results(infile):
    file = open(infile)
    lines = []
    for _line in file.readlines():
        line = _line.strip("\n").split("\t")
        lines.append((line[0], line[1], line[3]))
    file.close()
    return lines


def get_seq_less(train_fasta, test_fasta, seq_id=0.3):

    print("Number of sequences in train fasta: {}".format(count_fasta_sequences(train_fasta)))
    print("Number of sequences in test fasta: {}".format(count_fasta_sequences(test_fasta)))

    # make temporary directory
    wkdir = BASE_DATA_DIR + "/evaluation/seq_ID/{}".format(seq_id)
    create_directory(wkdir)


    print("Creating target Database")
    target_dbase = wkdir+"/target_dbase"
    CMD = "mmseqs createdb {} {}".format(train_fasta, target_dbase)
    subprocess.run(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    print("Creating query Database")
    query_dbase = wkdir+"/query_dbase"
    CMD = "mmseqs createdb {} {}".format(test_fasta, query_dbase)
    subprocess.run(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


    '''print("Run MMseqs2 Map")
    result_dbase = wkdir+"/result_dbase"
    CMD = "mmseqs map {} {} {} {} --min-seq-id {}".\
        format(query_dbase, target_dbase, result_dbase, wkdir, seq_id)
    subprocess.run(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


    print("Extract Alignments")
    final_res = wkdir+"/final_res.tsv"
    CMD = "mmseqs convertalis {} {} {} {}".\
        format(query_dbase, target_dbase, result_dbase, final_res)
    subprocess.run(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)'''


    print("Run MMseqs2 Search")
    result_dbase = wkdir+"/result_dbase"
    CMD = "mmseqs search {} {} {} {} --min-seq-id {}".\
        format(query_dbase, target_dbase, result_dbase, wkdir, seq_id)
    subprocess.run(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    print("Extract Alignments")
    final_res = wkdir+"/final_res.tsv"
    CMD = "mmseqs convertalis {} {} {} {}".\
        format(query_dbase, target_dbase, result_dbase, final_res)
    subprocess.run(CMD, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


    lines = extract_from_results(final_res)

    shutil.rmtree(wkdir)

    queries, targets, seq_ids = zip(*lines)

    queries = set(queries)
    targets = set(targets)

    test_proteins = set(get_all_sequences(test_fasta))

    result = test_proteins.difference(queries)

    print("Number of sequences in test fasta: {}".format(len(test_proteins)))
    print("Number of sequences in test fasta less than {}: {}".format(seq_id, len(result)))


    return result


def read_filter_write(proteins, in_file, out_file):

    with open(in_file, 'r') as infile, open(out_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('\t')
                
            protein_id, go_term, score = parts
            
            if protein_id in proteins:
                outfile.write(line)


def read_filter_write_gt(proteins, in_file, out_file):

    with open(in_file, 'r') as infile, open(out_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('\t')
                
            protein_id, go_term = parts
            
            if protein_id in proteins:
                outfile.write(line)



def main():

    train_fasta = BASE_DATA_DIR + "/cafa5/Train/train_sequences.fasta"
    test_fasta = BASE_DATA_DIR + "/test/raw/sequence2.fasta"

    methods = ['naive', 'diamondblast', 'deepgose', 'sprof', 'transfew', 'Sequence', 'Structure', 'Text', 'Interpro', 'Consensus_w_structure', 'Consensus_wo_structure']
    ontologies = ['CC', 'MF', 'BP']

    in_file_pths = BASE_DATA_DIR + "/evaluation/predictions/{}/{}.tsv"
    out_file_pths = BASE_DATA_DIR + "/evaluation/predictions_0.3/{}/{}.tsv"

    gt_in_file_pths = BASE_DATA_DIR + "/evaluation/groundtruth/{}.tsv"
    gt_out_file_pths = BASE_DATA_DIR + "/evaluation/groundtruth/{}_{}.tsv"

    seq_id = 0.3


    proteins_less_30 = get_seq_less(train_fasta=train_fasta, test_fasta=test_fasta, seq_id=seq_id)


    for ont in ontologies:
        read_filter_write_gt(proteins_less_30, gt_in_file_pths.format(ont), gt_out_file_pths.format(ont, seq_id))

        create_directory(BASE_DATA_DIR + "/evaluation/predictions_{}/{}".format(seq_id, ont))
        for method in methods:
            in_file = in_file_pths.format(ont, method)
            out_file = out_file_pths.format(ont, method)
            print("Ontology: {} --- Method {}".format(ont, method))
            read_filter_write(proteins_less_30, in_file, out_file)




if __name__ == '__main__':

    main()

    
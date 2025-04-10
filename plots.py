import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_processing.utils import load_pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import  AutoTokenizer



def collect_stats():
    data = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/data/raw_data")

    # Collect the number of GO terms for each protein
    go_counts = [len(data[protein]['GO Terms']) for protein in data]

    # Calculate statistics
    if go_counts:
        # Display basic statistics
        min_go_terms = min(go_counts)
        max_go_terms = max(go_counts)
        avg_go_terms = sum(go_counts) / len(go_counts)

        print(f"Number of GO terms per protein:")
        print(f"Minimum: {min_go_terms}")
        print(f"Maximum: {max_go_terms}")
        print(f"Average: {avg_go_terms:.2f}")

        # Define bins for distribution
        bins = range(min_go_terms, max_go_terms + 2)  # +2 to include the last bin
        counts, _ = np.histogram(go_counts, bins=bins)

        # Find the range with the most proteins
        max_count = max(counts)
        most_common_range = [(bins[i], bins[i + 1]) for i in range(len(counts)) if counts[i] == max_count]

        print(f"Most common range(s) of GO terms: {most_common_range} with {max_count} proteins")

        # Optional: Visualize the distribution
        plt.hist(go_counts, bins=bins, alpha=0.7, edgecolor='black')
        plt.title('Distribution of GO Terms per Protein')
        plt.xlabel('Number of GO Terms')
        plt.ylabel('Number of Proteins')
        # plt.xticks(bins)
        plt.grid(axis='y')
        plt.savefig("pet.jpg")
    else:
        print("No protein data available.")



def plot_sequence_size():

    data = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/raw/combined_data")
    ontology_data = load_pickle("/home/fbqc9/Workspace/MCLLM_DATA/DATA/data/ontology_list")


    model_name = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sequence_lengths_1 = []
    sequence_lengths_2 = []
    sequence_lengths_3 = []

    sequence_lengths_1 = [len(tokenizer(data[i]['Text'])['input_ids']) for i in data if data[i]['Text'] is not None]
    sequence_lengths_2 = [len(tokenizer(data[i]['Interpro'])['input_ids']) for i in data if data[i]['Interpro'] is not None]
    sequence_lengths_3 = [len(tokenizer(ontology_data[i])['input_ids']) for i in ontology_data]


    data = {
            "Length": sequence_lengths_1 + sequence_lengths_2 + sequence_lengths_3,
            "Dataset": ["Text"] * len(sequence_lengths_1) + 
                    ["Interpro"] * len(sequence_lengths_2) + 
                    ["Ontology"] * len(sequence_lengths_3)
            }

    palette = {"Text": "orange", "Interpro": "purple", "Ontology": "red"}
    df = pd.DataFrame(data)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="Dataset", y="Length", data=df, palette=palette)
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Dataset")
    plt.ylabel("Sequence Length")

    # Create custom patches for the legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=palette[key], label=key) for key in palette]
    ax.legend(handles=legend_elements, title="Dataset")

    plt.tight_layout()
    plt.savefig("box_plot.jpg")



plot_sequence_size()
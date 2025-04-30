import pandas as pd
import torch.nn.functional as F
from data_processing.utils import load_pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import  AutoTokenizer
from Const import BASE_DATA_DIR


def plot_sequence_size():

    data = load_pickle(BASE_DATA_DIR + "/data/raw/combined_data")
    ontology_data = load_pickle(BASE_DATA_DIR + "/data/ontology_list")


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
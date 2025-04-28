# FunBind: A multimodal foundational AI model for improving protein function prediction
[![View Preprint on bioRxiv](https://img.shields.io/badge/Preprint-bioRxiv-b31b1b)](https://github.com/jianlin-cheng/FunBind/blob/main/model.png)



`FunBind` is a multimodal model for protein function prediction, combining protein sequences, structures, textual descriptions, domain annotations, and ontology information.
It supports both direct `classification` and `zero-shot` prediction of novel functional terms through cross-modal contrastive learning.

![Method overview ](model.png)


## Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Inference](#inference)
    - [Zero-shot Inference](#zero-shot-inference)
    - [Classification](#function-classification)
- [Training](#training)
    - [Self- supPretraining](#how-to-run-inference-with-a-method-ensemble)
    - [Training](#how-to-create-comparative-plots-of-inference-results)
- [License](#license)
- [Citing this work](#reference)



## Installation

<details>
To get started, clone the repository and navigate to the project directory:
```
git clone https://github.com/jianlin-cheng/FunBind.git
cd FunBind
```

Then, create the environment and activate it by running:
```bash
conda env create -f FunBind.yml
conda activate FunBind
```
</details>


## Dataset 

<details>

</details>

## Inference
<details>



### Zero-shot Inference

Feature Extraction and Comparison Across Multiple Modalities (e.g., Sequence, Structure, Text, and Domain Annotations)



```python

from FunBind import data
import torch
from FunBind.models import funbind_model
from FunBind.models.funbind_model import ModalityType

# Example data for protein sequences, structures, text, and annotations
sequence_list = ["ATGCAGT", "GTCAGTAC", "CGTATCG"]
structure_paths = [".assets/structure1.pdb", ".assets/structure2.pdb", ".assets/structure3.pdb"]
text_paths = [".assets/text1.txt", ".assets/text2.txt", ".assets/text3.txt"]
annotations_paths = [".assets/annotations1.json", ".assets/annotations2.json", ".assets/annotations3.json"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = funbind_model.FunBind(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.SEQUENCE: data.load_and_transform_sequence(sequence_list, device),
    ModalityType.STRUCTURE: data.load_and_transform_structure(structure_paths, device),
    ModalityType.TEXT: data.load_and_transform_text(text_paths, device),
    ModalityType.ANNOTATION: data.load_and_transform_annotations(annotations_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print(
    "Sequence x Structure: ",
    torch.softmax(embeddings[ModalityType.SEQUENCE] @ embeddings[ModalityType.STRUCTURE].T, dim=-1),
)
print(
    "Sequence x Text: ",
    torch.softmax(embeddings[ModalityType.SEQUENCE] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Structure x Annotations: ",
    torch.softmax(embeddings[ModalityType.STRUCTURE] @ embeddings[ModalityType.ANNOTATION].T, dim=-1),
)

# Expected output:
#
# Sequence x Structure:
# tensor([[0.9876, 0.0012, 0.0112],
#         [0.0041, 0.9987, 0.0216],
#         [0.0105, 0.0167, 0.9934]])
#
# Sequence x Text:
# tensor([[0.9998, 0.0001, 0.0002],
#         [0.0003, 0.9996, 0.0007],
#         [0.0002, 0.0005, 0.9993]])
#
# Structure x Annotations:
# tensor([[0.9625, 0.0314, 0.0061],
#         [0.0457, 0.9342, 0.0201],
#         [0.0213, 0.0246, 0.9541]])

```


### Function Classification

```bash
python train.py --epochs [Number_epoch] --folder [intermediate_folder]
```

</details>

## Training
<details>

</details>

## Developer

<details></details>
```
Frimpong Boadu
Deparment of Computer Science
University of Missouri
Columbia, MO 65211, USA
Email: fbqc9@missouri.edu
```


## Contact
<details>
```
Jianlin (Jack) Cheng, PhD, AAAS Fellow
Curators' Distinguished Professor
William and Nancy Thompson Distinguished Professor
Department of Electrical Engineering and Computer Science
University of Missouri
Columbia, MO 65211, USA
Email: chengji@missouri.edu
```
</details>

## License
This project is covered under the MIT License

## Reference
<details>
FunBind: A multimodal foundational AI model for improving protein function prediction.
</details>
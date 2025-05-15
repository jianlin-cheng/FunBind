# FunBind: A unified multimodal model for generalizable zero-shot and supervised protein function prediction
[![View Preprint on bioRxiv](https://img.shields.io/badge/Preprint-bioRxiv-b31b1b)](https://www.biorxiv.org/content/10.1101/2025.05.09.653226v1#:~:text=Here,%20we%20present%20FunBind,%20a%20multimodal%20AI%20model,enhance%20prediction%20accuracy%20and%20infer%20previously%20unseen%20functions.)



`FunBind` is a multimodal model for protein function prediction, combining protein sequences, structures, textual descriptions, domain annotations, and ontology information.
It supports both direct `classification` and `zero-shot` prediction of novel functional terms through cross-modal contrastive learning.

![Method overview ](models/model.png)


## 🗊 Contents
<details>
<summary>Table of contents</summary>


- [Installation Instructions](#&#9881;-installation-instructions)
- [Dataset Format](#&#128218;-dataset-format)
    - [Sequence Data](#&#129516;-sequence-data)
    - [Structure Data](#&#129521;-structure-data)
    - [Text Data](#&#128221;-text-data)
    - [InterPro Data](#&#129513;-interpro-data)
    - [Ontology Data](#&#129504;-ontology-data)
- [Inference](#&#128640-inference)
    - [Zero-shot Inference](#zero-shot-inference)
    - [Classification](#supervised-classification)
- [Training](#&#128293;-training)
    - [Self-supervised Pretraining](#self-supervised-pretraining)
    - [Supervised Function Classification via Fine-Tuning](#supervised-function-classification-via-fine-tuning)
    - [Reproducing Experiments](#experiments-conducted)
- [Contributors](#&#128187;-contributors)
- [Contact](#&#128386;-contact)
- [License](#&#128274;-license)
- [Citing this work](#&#128214;-citing-this-work)

</details>


## &#9881; Installation Instructions
<details>
<summary>To get started with FunBind, follow these steps</summary>

1. Clone the Repository
```
git clone https://github.com/jianlin-cheng/FunBind.git
cd FunBind
```

2. Download checkpoints (~ 16GB total):
<!-- 
   wget url-for-data
   downloaded-data
-->
```
https://calla.rnet.missouri.edu/rnaminer/funbinddata/DATA/saved_models/
```

3. Set Up the Conda Environment:
```bash
conda env create -f FunBind.yml
conda activate FunBind
```
</details>


## &#128218; Dataset Format

<details>
<summary>Click to expand dataset format details</summary>

---

### &#129516; Sequence Data  
Sequences should be provided in **FASTA** format.  
**Example:** [`examples/sequence.fasta`](examples/sequence.fasta)

---

### &#129521; Structure Data  
Structure data can be obtained from **AlphaFold** and converted into **3Di sequences** using  
[**ProstT5 – How to derive 3Di sequences from structures**](https://github.com/mheinzinger/ProstT5?tab=readme-ov-file#-how-to-derive-3di-sequences-from-structures).  
The resulting **3Di FASTA** file can then be used as input to **FunBind**.  
**Example:** [`examples/structure.fasta`](examples/structure.fasta)

---

### &#128221; Text Data  
Text descriptions should follow the **[UniProt Flat Text format](http://web.expasy.org/docs/userman.html)**.  
You can download data using the [**UniProt ID Mapping Tool**](https://www.uniprot.org/id-mapping).  
**Example:** [`examples/text.txt`](examples/text.txt)

---

### &#129513; InterPro Data  
InterPro domain annotations can be generated using **[InterProScan](https://www.ebi.ac.uk/interpro/download/)**.  
**Example:** [`examples/text.txt`](examples/text.txt)

---

### &#129504; Ontology Data  
Ontology annotations (e.g., Gene Ontology terms) should be provided in a simple text format, where each line contains a **GO ID**.
**Example:** [`examples/text.txt`](examples/ontology.txt)

---

</details>


## &#128640; Inference
<details>
<summary>Run inference with pre-trained models.</summary>


### Zero-shot Inference

```bash
    python zeroshot_inference.py [-h] \
        --input-path INPUT_PATH \
        --modality {Sequence,Structure,Text,Interpro} \
        --ontology-path ONTOLOGY_PATH \
        --go-graph GO_GRAPH \
        --model-checkpoint MODEL_CHECKPOINT \
        [--batch BATCH] \
        [--topk TOPK] \
        [--device DEVICE]
```

####  Example:

To run zero-shot inference using Text modality on the sample data in the examples/ directory:

```bash
python zeroshot_inference.py \
    --model-checkpoint /path/to/funbind_checkpoint.pth \
    --input-path examples/text.txt \
    --modality Text \
    --ontology-path examples/ontology.txt \
    --go-graph examples/go-basic.obo
```

This will give you the output
```python
Predictions for protein: Q64565
Top 1 term: ('GO:0170035',), Score: 85.83%
Top 2 term: ('GO:0170033',), Score: 13.29%
Top 3 term: ('GO:1902674',), Score: 0.34%
-----------------------------
Predictions for protein: A8BPK8
Top 1 term: ('GO:1905504',), Score: 87.74%
Top 2 term: ('GO:0097561',), Score: 5.90%
Top 3 term: ('GO:0097560',), Score: 5.66%
-----------------------------
Predictions for protein: Q12198
Top 1 term: ('GO:0170043',), Score: 63.78%
Top 2 term: ('GO:0170033',), Score: 20.56%
Top 3 term: ('GO:0170041',), Score: 9.86%
-----------------------------
Predictions for protein: P18335
Top 1 term: ('GO:0170038',), Score: 95.72%
Top 2 term: ('GO:0170035',), Score: 3.15%
Top 3 term: ('GO:0170039',), Score: 1.13%
-----------------------------
```


### Supervised Classification

```bash
python train.py --epochs [Number_epoch] --folder [intermediate_folder]
```

</details>

## &#128293; Training

<details>
<summary>Instructions for training the model.</summary>

You can download the preprocessed training and validation data, as well as the data for experiments from (~36 GB total):
```
https://calla.rnet.missouri.edu/rnaminer/funbinddata
```

### Self-supervised Pretraining

1. To Train the model use the script:
```bash
python pretraining.py
```

### Supervised Function Classification via Fine-

1. To Train the model use the script:
```bash
python training.py
```


2. Evaluation command used: see 
see [cafa evaluator](https://github.com/BioComputingUP/CAFA-evaluator)
```bash
cafaeval obo-file-path predictions-path groundtruth-file -out_dir output-path -ia information-acretion-file -prop fill -norm cafa -th_step 0.001 -max_terms 500
```


</details>



## &#128187; Contributors
<details>
<summary>Click to expand</summary>

<p align="left">
  <a href="https://github.com/frimpz">
    <img src="https://github.com/frimpz.png" width="50" height="50" style="border-radius: 50%;" />
  </a>
  &nbsp;  &nbsp;  &nbsp;   &nbsp;  &nbsp;
  <a href="https://github.com/yw7bh">
    <img src="https://github.com/yw7bh.png" width="50" height="50" style="border-radius: 50%;" />
  </a>
  &nbsp;  &nbsp;  &nbsp;   &nbsp;  &nbsp;
  <a href="https://github.com/jianlin-cheng">
    <img src="https://github.com/jianlin-cheng.png" width="50" height="50" style="border-radius: 50%;" />
  </a>
</p>

<p align="left">
  <a href="https://github.com/frimpz">@frimpz</a>
  &nbsp;  &nbsp;  &nbsp;   &nbsp;  &nbsp;
  <a href="https://github.com/yw7bh">@yw7bh</a>
  &nbsp;  &nbsp;  &nbsp;   &nbsp;  &nbsp;
  <a href="https://github.com/jianlin-cheng">@jianlin-cheng</a>
</p>
</details>


## &#128386; Contact
<details>
<summary>Reach us for support or inquiries.</summary>

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

## &#128274; License
This project is covered under the MIT License

## &#128214; Citing this work
<details>
Boadu, F., Wang, Y., Cheng, J. A unified multimodal model for generalizable zero-shot and supervised protein function prediction. Submitted. 
</details>
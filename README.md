# FunBind: A unified multimodal model for generalizable zero-shot and supervised protein function prediction
[![View Preprint on bioRxiv](https://img.shields.io/badge/Preprint-bioRxiv-b31b1b)](https://www.biorxiv.org/content/10.1101/2025.05.09.653226v1#:~:text=Here,%20we%20present%20FunBind,%20a%20multimodal%20AI%20model,enhance%20prediction%20accuracy%20and%20infer%20previously%20unseen%20functions.)



`FunBind` is a multimodal model for protein function prediction, combining protein sequences, structures, textual descriptions, domain annotations, and ontology information.
It supports both direct `classification` and `zero-shot` prediction of novel functional terms through cross-modal contrastive learning.

![Method overview ](models/model.png)


## &#128458; Contents
<details>
<summary>Table of contents</summary>


- [&#9881;&#65039; Installation Instructions](#installation-instructions)
- [&#128218; Dataset Format](#dataset-format)
    - [&#129516; Sequence Data](#dataset-format)
    - [&#129521; Structure Data](#dataset-format)
    - [&#128221; Text Data](#dataset-format)
    - [&#129513; InterPro Data](#dataset-format)
    - [&#129504; Ontology Data](#dataset-format)
- [&#128640; Inference](#inference)
    - [Zero-shot Inference](#inference)
    - [Classification Inference](#inference) 
- [&#128293; Training](#training)
    - [Self-supervised Pretraining](#training)
    - [Supervised Function Classification via Fine-Tuning](#training)
    - [Reproducing Experiments](#training)
- [&#128187; Contributors](#contributors)
- [&#128386; Contact](#contact)
- [&#128274; License](#license)
- [&#128214; Citing this work](#citation)

</details>

<h2 id="installation-instructions"> &#9881;&#65039; Installation Instructions</h2>

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


<h2 id="dataset-format"> &#128218; Dataset Format</h2>

<details>
<summary>Click to expand dataset format details</summary>

---


<h3 id="sequence-data"> &#129516; Sequence Data</h3>

Sequences should be provided in **FASTA** format.  
**Example:** [`examples/sequence.fasta`](examples/sequence.fasta)

---

<h3 id="structure-data"> &#129521; Structure Data</h3>

Structure data can be obtained from **AlphaFold** and converted into **3Di sequences** using  
[**ProstT5 – How to derive 3Di sequences from structures**](https://github.com/mheinzinger/ProstT5?tab=readme-ov-file#-how-to-derive-3di-sequences-from-structures).  
The resulting **3Di FASTA** file can then be used as input to **FunBind**.  
**Example:** [`examples/structure.fasta`](examples/structure.fasta)

---

<h3 id="text-data"> &#128221; Text Data</h3>

Text descriptions should follow the **[UniProt Flat Text format](http://web.expasy.org/docs/userman.html)**.  
You can download data using the [**UniProt ID Mapping Tool**](https://www.uniprot.org/id-mapping).  
**Example:** [`examples/text.txt`](examples/text.txt)

---

<h3 id="interpro-data"> &#129513; InterPro Data</h3>

InterPro domain annotations can be generated using **[InterProScan](https://www.ebi.ac.uk/interpro/download/)**.  
**Example:** [`examples/text.txt`](examples/text.txt)

---

<h3 id="ontology-data"> &#129504; Ontology Data</h3>

 
Ontology annotations (e.g., Gene Ontology terms) should be provided in a simple text format, where each line contains a **GO ID**.  
**Example:** [`examples/ontology.txt`](examples/ontology.txt)

---

</details>


<h2 id="inference"> &#128640; Inference</h2>
<details>
<summary>Run inference with pre-trained models.</summary>


<h3 id="zero-shot-inference"> Zero-shot Inference</h3>

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



<h3 id="classification-inference"> Classification Inference</h3>


```bash
    python classification_inference.py [-h] 
      --data-path DATA_PATH 
      [--sequence-path SEQUENCE_PATH]
      [--structure-path STRUCTURE_PATH] 
      [--text-path TEXT_PATH]
      [--interpro-path INTERPRO_PATH] 
      [--ontology ONTOLOGY] 
      [--device DEVICE]
      [--num-batches NUM_BATCHES] 
      [--working-dir WORKING_DIR] 
      [--output OUTPUT]
```


####  Example:

To run classification using the sample data provided in the examples/ directory:"

```bash
python python classification_inference.py --sequence-path examples/sequence.fasta --structure-path examples/structure.fasta --data-path /home/fbqc9/Workspace/MCLLM_DATA/DATA/inference --device cuda:0
```


</details>


<h2 id="training"> &#128293; Training</h2>

<details>
<summary>Instructions for training the model.</summary>

You can download the preprocessed training and validation data, as well as the data for experiments from (~36 GB total):
```
https://calla.rnet.missouri.edu/rnaminer/funbinddata
```

<h3 id="self-supervised-pretraining"> Self-supervised Pretraining</h3>

1. To Train the model use the script:
```bash
python pretraining.py
```

<h3 id="supervised-classification"> Supervised Classification via Fine-tuning</h3>

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


<h2 id="contributors"> &#128187; Contributors </h2>
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


<h2 id="contact"> &#128386; Contact</h2>
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


<h2 id="license"> &#128274; License </h2>
This project is covered under the MIT License


<h2 id="citation"> &#128214;  Citing this work Inference</h2>
<details>
Boadu, F., Wang, Y., Cheng, J. A unified multimodal model for generalizable zero-shot and supervised protein function prediction. Submitted. 
</details>
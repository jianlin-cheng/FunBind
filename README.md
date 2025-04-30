# FunBind: A multimodal foundational AI model for improving protein function prediction
[![View Preprint on bioRxiv](https://img.shields.io/badge/Preprint-bioRxiv-b31b1b)](https://github.com/jianlin-cheng/FunBind/blob/main/model.png)



`FunBind` is a multimodal model for protein function prediction, combining protein sequences, structures, textual descriptions, domain annotations, and ontology information.
It supports both direct `classification` and `zero-shot` prediction of novel functional terms through cross-modal contrastive learning.

![Method overview ](models/model.png)


## Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Inference](#inference)
    - [Zero-shot Inference](#zero-shot-inference)
    - [Classification](#function-classification)
- [Training](#training)
    - [Self-supervised Pretraining](#self-supervised-pretraining)
    - [Supervised Function Classification via Fine-Tuning](#supervised-function-classification-via-fine-tuning)
- [License](#license)
- [Citing this work](#reference)



## Installation
#### To get started with FunBind, follow these steps:

<details>

1. Clone the Repository
```
git clone https://github.com/jianlin-cheng/FunBind.git
cd FunBind
```

2. Download checkpoints (~ GB total):
```
wget url-for-data
unzip downloaded-data
```

3. Set Up the Conda Environment:
```bash
conda env create -f FunBind.yml
conda activate FunBind
```
</details>


## Dataset 

<details>

Sample Sequence Data:
```
Sequence data can be provided in the fasat format. See [`examples/sequence.fasta`](examples/text.txt).
```

Sample Structure Data:
```
Structure Data can be downloaded from Alphafold, and converted to 3DI sequences. see [](https://github.com/mheinzinger/ProstT5).
```

Sample Text Data:
```
The text data can be provided in the [UniProt Flat Text format]
(https://www.uniprot.org/help/uniprotkb_format). You can 
download data in this format using the [UniProt ID Mapping 
tool](https://www.uniprot.org/id-mapping). For an example of
 the expected format, please refer to the file located at
  [`examples/text.txt`](examples/text.txt).
```

Sample Interpro Data:
```
Interpro data is can be downloaded or generated with Interproscan [Interpro](https://www.ebi.ac.uk/interpro/download/)
```

Sample Ontology Data:
```

```

</details>

## Inference
<details>



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


### Function Classification

```bash
python train.py --epochs [Number_epoch] --folder [intermediate_folder]
```

</details>

## Training
<details>

### Self-supervised Pretraining

### Supervised Function Classification via Fine-Tuning

</details>

## Developer

<details>

```
Frimpong Boadu
Deparment of Computer Science
University of Missouri
Columbia, MO 65211, USA
Email: fbqc9@missouri.edu
```
</details>


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
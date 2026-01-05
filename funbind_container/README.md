# FunBind Container for Protein Function Prediction

This Docker container implements a multi-modality based FunBind protein function prediction pipeline. `FunBind` is a multimodal model for protein function prediction, combining protein sequences, structures, textual descriptions, domain annotations, and ontology information through cross-modal contrastive learning.

## &#128458; Contents
- `funbind_main.py` - Main orchestration script
- `run_funbind.sh`  - shell scriot for antomatically active environment and excuate main orchestration script **funbind_main.py**
- `download_model.sh` - shell script for automatically downloading model weights
- `get_pretrainedModels.py` - script for downloading all pretrained models 
- `extract_data.py` - extract data info from each modality and combine them
- `dataset.py` - Customized dataloader
- `model.py` - Funbind main model
- `utils.py` - data processing utilities
- `config.yaml` - Funbind model configuration
- `requirements.txt` - Python dependencies
- `entry.list` - Interpro entry info
- `ParentChildTreeFile.txt` - Intepro entry relationship
- `Dockerfile` - Container definition

## &#9881;&#65039; How It Works
**Step 1**: using `download_model.sh` download all the needed weights for diferent models to avoid re-downloading models.

**Step 2**: using `funbind_main.py` to complete the prediction, it orchestrates three stages to complete protein function prediction. 


- 1. **FunBind Embeddings**: The container first extract feature embeddings from the pretrained models and saved in a customized folder provided by user!
- 2. **FunBind Predictions**: Predict CC, MF, BP three kind terms one by one so that user can get each invidual one and combined one!
- 3. **Save Predictions**: Save the predicted results into a customized folder provided by user.


## &#128218; Input Dataset Format for each modality (Best single modality:Text) 

- &#129516; Sequence Modality

  Sequences should be provided in **FASTA** format, an example can be found at your cache-path: $PWD/checkpoints/examples/classification.
**Example:** [`$PWD/checkpoints/examples/classification/sequence.fasta`]($PWD/checkpoints/examples/classification/sequence.fasta)

- &#129521; Structure Modality

  Structure data can be obtained from **AlphaFold** and converted into **3Di sequences** using  
[**ProstT5 – How to derive 3Di sequences from structures**](https://github.com/mheinzinger/ProstT5?tab=readme-ov-file#-how-to-derive-3di-sequences-from-structures).  
The resulting **3Di FASTA** file can then be used as input to **FunBind**, an example can be found at your cache-path: $PWD/checkpoints/examples/classification..  
**Example:** [$PWD/checkpoints/examples/classification/structure.fasta`]($PWD/checkpoints/examples/classification/structure.fasta)


- &#128221; Text Modality (Best single Modality)

  Text descriptions should follow the **[UniProt Flat Text format](http://web.expasy.org/docs/userman.html)**.  
You can download data as text format using the [**UniProt ID Mrooting Tool**](https://www.uniprot.org/id-mrooting). An example can be found at your cache-path: $PWD/checkpoints/examples/classification.  
**Example:** [`$PWD/checkpoints/examples/classification/text.txt`]($PWD/checkpoints/examples/classification/text.txt)

- &#129513; InterPro Modality

  InterPro domain annotations can be generated using **[InterProScan](https://www.ebi.ac.uk/interpro/download/)**. An example can be found at your cache-path: $PWD/checkpoints/examples/classification.  
**Example:** [`$PWD/checkpoints/examples/classification/interpro.txt`]($PWD/checkpoints/examples/classification/interpro.txt)

## &#128640; Model caching

The `download_model.sh` script is used to download and cache all required model weights and metadata before running the container, ensuring they are available for prediction.

```bash
# The pretrained_funBind_model will be automatically downloaded to $PWD/checkpoints, while the pretrained_huggingface_models will be downloaded to the cache path specified by the user.

bash ./download_model.sh --cache-path /path/pretrained_huggingface_models/cache_folder --token hf_xxx
```


#### Notes: following bellow instruction to complete your download!
❌ Hugging Face token required to download LLaMA 2 when using text/interpro modality.
- 👉 You must request access before downloading:

  - 1️⃣ Go to Hugging Face [Request LLaMA 2 access](https://huggingface.co/meta-llama)
  - 2️⃣ Select your model (e.g., meta-llama/Llama-2-7b-hf)
  - 3️⃣ Click **"Access requested"** and wait for rootroval
  - 4️⃣ [Create your Hugging Face token](https://huggingface.co/settings/tokens)
  - 5️⃣ Re-run bash script with `--token hf_xxx`


## &#128293; Building the Container
The container should be built after all scripts have been generated/collected. 

```bash
# From the funbind_container directory
docker build --network=host -t funbind_predictor .

```

## &#128293; Running the Container with Model Caching


```bash
docker run --rm \
  --name funbind_predictor \
  -v /path/to/test_data:/root/data \
  -v /path/to/test_output:/root/output \
  -v /path/pretrained_huggingface_models/cache_folder:/root/.cache/huggingface/hub \
  -v /path/to/pretrained_funbind_model:/root/.cache/checkpoints \
  funbind_predictor \
  --data-path /root/.cache/checkpoints \
  --sequence-path /root/data/sequence.fasta \
  --structure-path /root/data/structure.fasta \
  --interpro-path /root/data/interpro.txt \
  --text-path /root/data/text.txt \
  --output /root/data \
  --base-model prostt5
```
**Notes:**
- Any Hugging Face model weights downloaded on the local server must be mounted to `/root/.cache/huggingface/hub` inside the Docker container. FunBind model weights can be mounted anywhere. Otherwise, prediction will fail.

####  Required Arguments

- `--data-path`: Path to the mounted pretrained Funbind model checkpoints inside the container, which automatically contains different version funbind models & Go term list & other relevant data.
- `--output /root/data`: Path to output folder, which will contains all individual modality and term prediction and a final combined one across three terms and modalities you provided.
- `--sequence-path`: Path to the inpuf file(sequence), a FASTA file containing query sequences to predict.

#### Optional Arguments
- `--structure-path`: Path to the inpuf file(structure), a FASTA file containing 3di sequence extracted from alphfold predicted structure.
- `--interpro-path`: Path to the input file(Interpro).
- `--text-path`: Path to the input file(Text).
- `--base-model`: choicing in ['esm2', 'prostt5'], default='prostt5'

## &#128187; GPU Requirements

- **Recommended**: A100/H100
- **Minimum**: TBD
- **CPU fallback**: Will run on CPU but significantly slower

## &#128187; Model Information

- **Default Model**: `/root/.cache/checkpoints`
- **Model Size**: rootroximately 90 GB for all required Hugging Face pretrained models, plus ~22 GB for the various FunBind pretrained models.
- **Cache Location**: `/root/.cache/huggingface/hub` for all needed huggingface pretrained models, `/root/.cache/checkpoints` for funbind pretrained model along with its required info (mountable).
- **First build and rebuild**: All required model weights and metadata will be downloaded and cached up front, reducing both the initial build time and any future rebuild times.

## &#128187; Dependencies

- **CUDA**: 12.1+ (for GPU support)
- **Python**: 3.10+
- **PyTorch**: 2.4.0+ with CUDA support
- **Transformers**: 4.44.2 (HuggingFace)
- **BioPython**: 1.84 (or sequence parsing)
- **scikit-learn**: For similarity calculations
- **NetworkX and obonet**: For ontology processing
- **SciPy**: For sparse matrix operations
- **Pandas**: For data manipulation

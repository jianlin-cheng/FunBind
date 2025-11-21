r'''
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, EsmTokenizer, EsmModel
def get_modelweights_offline(args):
    cache_dir = args.cache_path
    model_map = {
        "esm2_t48": ('facebook/esm2_t48_15B_UR50D', EsmTokenizer, EsmModel),
        "prostt5": ('Rostlab/ProstT5', T5Tokenizer, AutoModel),
        "llama2": ('meta-llama/Llama-2-7b-hf', AutoTokenizer, AutoModel)
    }

    for model_name, (model_path, tokenizer_class, model_class) in model_map.items():
        print(f"⬇️ Downloading tokenizer for {model_name}...")
        if 'prostt5' in model_name:
            tokenizer_class.from_pretrained(model_path, cache_dir=cache_dir, do_lower_case=False)
        else:
            tokenizer_class.from_pretrained(model_path, cache_dir=cache_dir)

        print(f"⬇️ Downloading model weights for {model_name}...")
        model_class.from_pretrained(model_path, cache_dir=cache_dir)

    print("✅ All models and tokenizers downloaded to cache.")
'''

from huggingface_hub import snapshot_download

def download_all_models(args):
    save_dir = args.cache_path
    model_repos = {
        "esm2_t48": "facebook/esm2_t48_15B_UR50D",
        "prostt5": "Rostlab/ProstT5",
        "llama2": "meta-llama/Llama-2-7b-hf",
    }

    for name, repo in model_repos.items():
        print(f"⬇️ Downloading {name} ({repo}) ...")

        snapshot_download(
            repo_id=repo,
            local_dir=f"{save_dir}/{name}",
            local_dir_use_symlinks=False   # ensure REAL files, not symlinks
        )

        print(f"✅ Finished {name}")

    print("🎉 All model files downloaded!")

   

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download pretrained models for offline use.")

    parser.add_argument('--cache-path', type=str, default="./hub", help="Path to the save the download pretrained model")
    parser.add_argument('--token', type=str, default=None,  help="token for your login huggingface account to use text or interpro modality")

    args = parser.parse_args()
   
    ## args.cache_path = "/bml/LAFA/FunBind/funbind_container/hub"

    
    if not args.token:
        raise ValueError(
            "❌ Hugging Face token required to download LLaMA 2 when using text/interpro modality.\n"
            "👉 You must first request access on Hugging Face before using this modality by following the steps below:\n"
            "   1️⃣ Go to: https://huggingface.co/meta-llama\n"
            "   2️⃣ Choose your model (e.g., meta-llama/Llama-2-7b-hf)\n"
            "   3️⃣ Click 'Access requested' and wait for approval\n"
            "   4️⃣ Then create a token at: https://huggingface.co/settings/tokens\n"
            "   5️⃣ Run again with `--token hf_xxx`\n"
        )
    from huggingface_hub import login
    try:
        login(token=args.token)  # automatically saved at: ~/.cache/huggingface/hub  
    except Exception as e:
        raise RuntimeError(f"❌ Failed to authenticate with Hugging Face token: {e}")
    
    # get_modelweights_offline(args)
    download_all_models(args)
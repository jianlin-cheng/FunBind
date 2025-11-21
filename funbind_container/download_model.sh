#!/bin/bash
set -e

# usage  if cache_path is not provided, it will be saved in the default path: ~/.cache/huggingface/hub
# bash ./download_model.sh --cache-path /home/user/hf_cache --token hf_xxx

# 0 Download FunBind model weights. Indenting the second line (by 4 spaces or a tab) is just a visual cue
wget -r -np -nH --cut-dirs=4 -R "index.html*" -nc -P "$PWD/checkpoints" \
    https://calla.rnet.missouri.edu/rnaminer/funbinddata/DATA/data/

echo "✅ FunBind Model download completed and saved at $PWD/checkpoints."

# 1 Create a virtual environment if it doesn't exist
if [ ! -d "hf_download" ]; then
    python3 -m venv hf_download
fi

# 2️ Activate the virtual environment
source hf_download/bin/activate

# 3️ Upgrade pip and install required packages
pip install --upgrade pip
pip install huggingface_hub

# 4️ Run the Python download script with all passed arguments
python get_pretrainedModels.py "$@"

# 5️ Deactivate the virtual environment
deactivate

# 6 Remove virtual environment
rm -rf hf_download

# 7 Check where the pretrained models are
DEFAULT_CACHE="$PWD/hub"

# Default value
CACHE_PATH=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --cache-path)
            CACHE_PATH="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            shift
            ;;
    esac
done

if [ -n "$CACHE_PATH" ] && [ -d "$CACHE_PATH" ] && [ "$(find "$CACHE_PATH" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "✅ Pretrained models download complete. Cache saved at your provided path: $CACHE_PATH"
else
    echo "✅ Pretrained models download complete. No custom cache path provided, so default location is used: $DEFAULT_CACHE"
fi

echo "💡 Reminder: your models and tokenizers are stored in the cache directory."
echo "✅ you can find them here:"
if [ -n "$CACHE_PATH" ] && [ -d "$CACHE_PATH" ] && [ "$(find "$CACHE_PATH" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "    $CACHE_PATH/esm2_t48/"
    echo "    $CACHE_PATH/prostt5/"
    echo "    $CACHE_PATH/llama2/"
else
    echo "    $DEFAULT_CACHE/esm2_t48/"
    echo "    $DEFAULT_CACHE/prostt5/"
    echo "    $DEFAULT_CACHE/llama2/"
fi

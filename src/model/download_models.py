import os
from huggingface_hub import snapshot_download, login

RESPONSE_PATH = "/cowrie/cowrie-git/src/model"

#MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

def download_model(model_name):
    """Download a Hugging Face model and tokenizer to the specified directory"""
    with open(f"{RESPONSE_PATH}/token.txt", "r") as f:
        token = f.read().rstrip()

    login(token=token)

    snapshot_download(repo_id=model_name, token=token)

if os.environ["COWRIE_USE_LLM"].lower() == "true":
    download_model(MODEL_NAME)
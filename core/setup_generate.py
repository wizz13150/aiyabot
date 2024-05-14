import urllib.request
import os

# Download first model : WizzGPTv2
model_folder_1 = "core/WizzGPTv2"
os.makedirs(model_folder_1, exist_ok=True)

base_url_1 = "https://huggingface.co/Wizz13150/WizzGPTv2/resolve/main/"

files_to_download_1 = ["config.json", "generation_config.json", "merges.txt", "pytorch_model.bin", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]

for file in files_to_download_1:
    filepath = os.path.join(model_folder_1, file)
    if not os.path.isfile(filepath):
        print(f"Missing file for WizzGPTv2. Downloading {file}")
        urllib.request.urlretrieve(base_url_1 + file, filepath)

# Download second model : DistilGPT2-Stable-Diffusion-V2
model_folder_2 = "core/DistilGPT2-Stable-Diffusion-V2"
os.makedirs(model_folder_2, exist_ok=True)

base_url_2 = "https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2/resolve/main/"

files_to_download_2 = ["config.json", "pytorch_model.bin", "tokenizer.json", "training_args.bin"]

for file in files_to_download_2:
    filepath = os.path.join(model_folder_2, file)
    if not os.path.isfile(filepath):
        print(f"Missing file for DistilGPT2-Stable-Diffusion-V2. Downloading {file}")
        urllib.request.urlretrieve(base_url_2 + file, filepath)

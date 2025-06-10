import requests
import os
from tqdm import tqdm

def download_with_tqdm(url, filepath):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    if total == 0:
        print(f"Downloading {os.path.basename(filepath)}…")
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(1024):
                if chunk:
                    file.write(chunk)
        return
    with open(filepath, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(filepath)}",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        leave=True
    ) as bar:
        for data in response.iter_content(1024):
            size = file.write(data)
            bar.update(size)

# ----------- WizzGPTv6 -----------
model_folder_1 = "core/WizzGPT6"
os.makedirs(model_folder_1, exist_ok=True)
base_url_1 = "https://huggingface.co/Wizz13150/WizzGPTv6/resolve/main/"
files_to_download_1 = [
    "config.json", "generation_config.json", "merges.txt", "pytorch_model.bin",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "WizzGPTv6.Q8_0.gguf"
]
for file in files_to_download_1:
    filepath = os.path.join(model_folder_1, file)
    if not os.path.isfile(filepath):
        print(f"Missing file for WizzGPTv6: {file}")
        url = base_url_1 + file
        download_with_tqdm(url, filepath)

# ----- Llama-3.2-11B-Vision-Instruct GGUF -----
model_folder_2 = "core/Llama-3.2-11B-Vision-Instruct-gguf"
os.makedirs(model_folder_2, exist_ok=True)
base_url_2 = "https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/resolve/main/"
files_to_download_2 = ["Llama-3.2-11B-Vision-Instruct.Q4_K_M.gguf"]
for file in files_to_download_2:
    filepath = os.path.join(model_folder_2, file)
    if not os.path.isfile(filepath):
        print(f"Missing file for Llama-3.2-11B-Vision-Instruct-gguf: {file}")
        url = base_url_2 + file
        download_with_tqdm(url, filepath)

for file in files_to_download_2:
    filepath = os.path.join(model_folder_2, file)
    if not os.path.isfile(filepath):
        print(f"Downloading: {file}")
        url = base_url_2 + file
        download_with_tqdm(url, filepath)
    else:
        print(f"Already exists: {file}")

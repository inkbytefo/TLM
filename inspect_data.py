import os
import requests

url = 'https://storage.googleapis.com/long-range-arena/lra_release/lra_release/listops-1000/basic_train.tsv'
file_path = 'data/lra_listops/train.tsv'

if not os.path.exists(file_path):
    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    resp = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(resp.content)

with open(file_path, 'r', encoding='utf-8') as f:
    for i in range(5):
        print(repr(f.readline()))

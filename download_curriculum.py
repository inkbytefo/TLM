import os
import gc
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_and_save(dataset_name, config_name, output_filename, max_samples, column_name="text"):
    """Downloads and saves a dataset with explicit cleanup."""
    try:
        print(f"\nDownloading {dataset_name} ({config_name if config_name else 'default'})...")
        # Load dataset
        if config_name:
            ds = load_dataset(dataset_name, config_name, split="train", streaming=True)
        else:
            ds = load_dataset(dataset_name, split="train", streaming=True)
            
        # Take samples
        ds_head = ds.take(max_samples)
        
        output_path = os.path.join(DATA_DIR, output_filename)
        print(f"Saving to {output_path}...")
        
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in tqdm(ds_head, total=max_samples):
                text = sample.get(column_name, "")
                if text:
                    f.write(text + "\n<|endoftext|>\n")
                    count += 1
        
        print(f"Saved {count} samples.")
        
        # Explicit cleanup to avoid threading issues at exit
        del ds_head
        del ds
        gc.collect()
        
    except Exception as e:
        print(f"Failed to download {dataset_name}: {e}")

def main():
    print("Downloading ASI Curriculum Datasets...")

    # --- PHASE 1: MORPHOLOGY & LOGIC ---
    download_and_save("wikimedia/wikipedia", "20231101.tr", "turkish_academic.txt", 20000, "text")
    download_and_save("bigcode/the-stack-smol", "default", "github_code.txt", 10000, "content")

    # --- PHASE 2: WORLD KNOWLEDGE ---
    download_and_save("roneneldan/TinyStories", None, "english_pile.txt", 20000, "text")

    # --- PHASE 3: REASONING ---
    download_and_save("nampdn-ai/tiny-textbooks", None, "math_reasoning.txt", 5000, "text")

    print("\nDownload Complete! Datasets are ready in 'data/' folder.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

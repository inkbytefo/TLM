import os
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def save_dataset_to_text(dataset, output_path, column_name="text", max_samples=10000):
    """Saves a Hugging Face dataset to a text file."""
    print(f"Saving {max_samples} samples to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for sample in tqdm(dataset):
            text = sample.get(column_name, "")
            if text:
                f.write(text + "\n<|endoftext|>\n")
                count += 1
                if count >= max_samples:
                    break
    print(f"Saved {count} samples.")

def main():
    print("Downloading ASI Curriculum Datasets...")

    # --- PHASE 1: MORPHOLOGY & LOGIC (Turkish + Code) ---
    
    # 1. Turkish Academic (Wikimedia/Wikipedia)
    try:
        print("\n[Phase 1] Downloading Turkish Wikipedia (wikimedia/wikipedia)...")
        # New standard wikipedia dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
        save_dataset_to_text(ds, os.path.join(DATA_DIR, "turkish_academic.txt"), column_name="text", max_samples=20000)
    except Exception as e:
        print(f"Failed to download Turkish Wiki: {e}")

    # 2. Code (The Stack Smol)
    try:
        print("\n[Phase 1] Downloading Code (bigcode/the-stack-smol)...")
        # 'default' contains mixed code, which is fine.
        ds = load_dataset("bigcode/the-stack-smol", "default", split="train", streaming=True)
        save_dataset_to_text(ds, os.path.join(DATA_DIR, "github_code.txt"), column_name="content", max_samples=10000)
    except Exception as e:
        print(f"Failed to download Code: {e}")

    # --- PHASE 2: WORLD KNOWLEDGE (English) ---
    
    try:
        print("\n[Phase 2] Downloading English (TinyStories)...")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        save_dataset_to_text(ds, os.path.join(DATA_DIR, "english_pile.txt"), column_name="text", max_samples=20000)
    except Exception as e:
        print(f"Failed to download English Data: {e}")

    # --- PHASE 3: REASONING (Math/Textbooks) ---
    
    try:
        print("\n[Phase 3] Downloading Reasoning (Tiny Textbooks)...")
        # Tiny Textbooks is excellent for reasoning
        ds = load_dataset("nampdn-ai/tiny-textbooks", split="train", streaming=True)
        save_dataset_to_text(ds, os.path.join(DATA_DIR, "math_reasoning.txt"), column_name="text", max_samples=5000)
    except Exception as e:
        print(f"Failed to download Reasoning Data: {e}")

    print("\nDownload Complete! Datasets are ready in 'data/' folder.")

if __name__ == "__main__":
    main()

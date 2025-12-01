import os
import numpy as np
from src.data.lra import load_listops_data

# Create dummy data
os.makedirs('data/lra_listops', exist_ok=True)
with open('data/lra_listops/train.tsv', 'w', encoding='utf-8') as f:
    f.write("Header\n")
    f.write("( ( 1 2 ) ) \t 3\n") # Text \t Label
    f.write("4 \t ( 5 6 )\n")     # Label \t Text (should handle both if I implemented correctly? No, I implemented priority)

# My implementation tries int(parts[0]) first.
# If "4" is first, it takes 4 as label.
# If "( ( 1 2 ) )" is first, int() fails, so it takes parts[1] "3" as label.

try:
    inputs, targets = load_listops_data('train', seq_len=10)
    print("Successfully loaded data!")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Targets: {targets}")
except Exception as e:
    print(f"Failed to load data: {e}")

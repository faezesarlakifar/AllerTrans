"""
End-to-end AllerTrans pipeline:
1. Generate ProtT5 embeddings (HDF5)
2. Generate ESM-2 embeddings (.pt per protein)
3. Run ensemble prediction
4. Save CSV results
"""

import subprocess
import os
import torch
import pandas as pd
from pathlib import Path
from allertrans.model import load_models, predict_ensemble
from allertrans.utils import load_h5_embeddings, load_esm_embeddings

# Paths
fasta_file = "examples/protein_sequences.fasta"
output_csv = "examples/predictions.csv"
temp_dir = Path("temp_embeddings")
temp_dir.mkdir(exist_ok=True)

protT5_h5 = temp_dir / "protT5_embeddings.h5"
esm_folder = temp_dir / "esm_embeddings"
esm_folder.mkdir(exist_ok=True)

# Step 1: Generate ProtT5 embeddings
print("Generating ProtT5 embeddings...")
subprocess.run([
    "python", "prott5_embedder.py",
    "--input", fasta_file,
    "--output", str(protT5_h5),
    "--per_protein", "1"
], check=True)

# Step 2: Generate ESM-2 embeddings
print("Generating ESM-2 embeddings...")
subprocess.run([
    "python", "extract.py",
    "esm2_t33_650M_UR50D", fasta_file, str(esm_folder),
    "--repr_layers", "0", "32", "33",
    "--include", "mean", "per_tok"
], check=True)

# Step 3: Load embeddings
print("Loading embeddings...")
protT5_emb, protT5_ids = load_h5_embeddings(protT5_h5)
esm_emb, esm_ids, _ = load_esm_embeddings(fasta_file, esm_folder)

# Align protein IDs by order
min_len = min(len(protT5_ids), len(esm_ids))
protT5_emb = protT5_emb[:min_len]
esm_emb = esm_emb[:min_len]
protein_ids = protT5_ids[:min_len]

# Step 4: Run ensemble prediction
model_protT5, model_cat = load_models()
concat_emb = torch.cat((esm_emb, protT5_emb), dim=1)
predictions = []

for i in range(min_len):
    pred = predict_ensemble(
        protT5_emb[i].unsqueeze(0),
        concat_emb[i].unsqueeze(0),
        model_protT5,
        model_cat
    )
    label = ["Non-Allergen", "Potential Allergen"][pred.item()]
    predictions.append({"id": protein_ids[i], "prediction": label})

# Step 5: Save results
df = pd.DataFrame(predictions)
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")

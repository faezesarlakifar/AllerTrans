import click
import subprocess
from pathlib import Path
import pandas as pd
import torch
from allertrans.model import load_models, predict_ensemble
from allertrans.utils import load_h5_embeddings, load_esm_embeddings

model_protT5, model_cat = load_models()

@click.group()
def run_cli():
    """AllerTrans CLI — Predict protein allergenicity from FASTA files."""
    pass

@run_cli.command("infer")
@click.option("--fasta", "-f", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path())
@click.option("--temp_dir", "-t", default="temp_embeddings", type=click.Path())
def infer(fasta, output, temp_dir):
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(exist_ok=True)

    protT5_h5 = temp_dir / "protT5_embeddings.h5"
    esm_folder = temp_dir / "esm_embeddings"
    esm_folder.mkdir(exist_ok=True)

    click.echo("Generating ProtT5 embeddings...")
    subprocess.run([
        "python", "prott5_embedder.py",
        "--input", fasta,
        "--output", str(protT5_h5),
        "--per_protein", "1"
    ], check=True)

    click.echo("Generating ESM-2 embeddings...")
    subprocess.run([
        "python", "extract.py",
        "esm2_t33_650M_UR50D", fasta, str(esm_folder),
        "--repr_layers", "0", "32", "33",
        "--include", "mean", "per_tok"
    ], check=True)

    click.echo("Loading embeddings...")
    protT5_emb, protT5_ids = load_h5_embeddings(protT5_h5)
    esm_emb, esm_ids, _ = load_esm_embeddings(fasta, esm_folder)

    if protT5_ids != esm_ids:
        click.echo("Warning: Protein IDs do not match. Aligning by order...")
        min_len = min(len(protT5_ids), len(esm_ids))
        protT5_emb = protT5_emb[:min_len]
        esm_emb = esm_emb[:min_len]
        protein_ids = protT5_ids[:min_len]
    else:
        protein_ids = protT5_ids

    concat_emb = torch.cat((esm_emb, protT5_emb), dim=1)
    predictions = []

    for i in range(len(protein_ids)):
        pred = predict_ensemble(
            protT5_emb[i].unsqueeze(0),
            concat_emb[i].unsqueeze(0),
            model_protT5,
            model_cat
        )
        label = ["Non-Allergen", "Potential Allergen"][pred.item()]
        predictions.append({"id": protein_ids[i], "prediction": label})

    df = pd.DataFrame(predictions)
    if output:
        df.to_csv(output, index=False)
        click.echo(f"Prediction saved to {output}")
    else:
        click.echo(df.to_string(index=False))
# allertrans/utils.py
import h5py
import torch
import click

def load_h5_embeddings(h5_path):
    """
    Load all embeddings from an H5 file (ProtT5) into a tensor and return protein IDs.
    """
    embeddings = []
    ids = []
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            try:
                if "representations" in f[key] and "mean" in f[key]["representations"]:
                    emb = f[key]["representations"]["mean"][:]
                else:
                    emb = f[key][()]
                embeddings.append(torch.tensor(emb).float())
                ids.append(key)
            except Exception as e:
                try:
                    click.echo(f"Skipping {key}: {e}")
                except:
                    print(f"Skipping {key}: {e}")
    if embeddings:
        return torch.stack(embeddings), ids
    else:
        return torch.empty((0, 0)), []
    
import os
import torch
from pathlib import Path
from esm import FastaBatchedDataset

def load_esm_embeddings(fasta_path, emb_path, label=None):
    """
    Load ESM embeddings using the exact procedure from extract.py.

    Args:
        fasta_path (str or Path): path to the input FASTA file
        emb_path (str or Path): folder with .pt embeddings
        repr_layer (int): which layer to extract (default 33)
        label (any, optional): optional label (not used for filenames)

    Returns:
        Xs (torch.Tensor): [num_proteins, embedding_dim]
        ids (list): protein IDs (labels)
        ys (list): optional labels if provided
    """
    fasta_path = Path(fasta_path)
    emb_path = Path(emb_path)

    # Load the FASTA using FastaBatchedDataset to get the exact labels
    dataset = FastaBatchedDataset.from_file(fasta_path)
    ids = [label for label, _ in dataset]
    Xs = []
    ys = []

    for label, _seq in dataset:
        fn = emb_path / f"{label}.pt"  # exactly like extract.py
        try:
            embs = torch.load(fn)
            repr_layer = 33
            Xs.append(embs['mean_representations'][repr_layer].float())
            if label is not None:
                ys.append(label)
        except Exception as e:
            print(f"Skipping {label}: {e}")

    if Xs:
        Xs = torch.stack(Xs, dim=0)
    else:
        Xs = torch.empty((0, 0))

    return Xs, ids, ys

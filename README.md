[![AllerTrans](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online%20Prediction%20Tool-blueviolet)](https://huggingface.co/spaces/sfaezella/AllerTrans)
[![AllerTrans](https://img.shields.io/badge/Publication-DOI:10.1093/biomethods/bpaf040-red)](https://doi.org/10.1093/biomethods/bpaf040)
[![Code Ocean](https://img.shields.io/badge/Code%20Ocean-Open%20Capsule-blue)](https://doi.org/10.24433/CO.1381053.v1)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)


<h1 align="center">
  AllerTrans
</h1>
<h2 align="center">
  A Deep Learning Method for Predicting the Allergenicity of Protein Sequences
</h2>

## Overview
Allergens are a major concern in protein safety, especially with the growing use of recombinant proteins in medical products. Traditional allergenicity tests are costly and time-consuming, prompting the need for efficient bioinformatics solutions. In this study, we developed an enhanced deep learning model that classifies proteins as allergenic or non-allergenic based on their sequences. Our method extracts features using two protein language models and combines them in a deep neural network, followed by ensemble modeling to improve performance. The proposed model achieved strong results: 97.91% sensitivity, 97.69% specificity, 97.80% accuracy, and a 99% AUC using five-fold cross-validation.

DOI: [https://doi.org/10.1093/biomethods/bpaf040](https://doi.org/10.1093/biomethods/bpaf040)

## Online Prediction Tool
You can try out the AllerTrans model directly available on Hugging Face Spaces:
[https://huggingface.co/spaces/sfaezella/AllerTrans](https://huggingface.co/spaces/sfaezella/AllerTrans)

<h2 align="center">
  A comprehensive flowchart that includes all of our experiments
</h2>

![Experiments' Flowchart](images/flowchart.jpg)

## Repository Structure
For transparency, this repository includes all the experiments, feature extraction, modeling notebooks, and tools necessary to reproduce the AllerTrans workflow.

```
AllerTrans/
├── notebooks/                  # All Jupyter notebooks, organized by workflow
│   ├── feature-extraction/     # Notebooks for extracting protein feature vectors
│   │   ├── 1.ESM-v2-embeddings.ipynb
│   │   ├── 2.ProtT5-embeddings.ipynb
│   │   └── 3.AAC-feature-vectors.ipynb
│   ├── modeling/               # Notebooks for training and evaluating models
│   │   ├── 1D-CNN.ipynb
│   │   ├── classic-machine-learning.ipynb
│   │   ├── nonlinear-DNN.ipynb
│   │   └── single-layer-LSTM.ipynb
│   └── additional-experiments/ # Supplementary experiments
├── src/                        # Users can run predictions on their own protein sequences in FASTA format via a single command.
│   ├── allertrans/             # CLI and scripts for end-to-end inference. 
│   │   ├── __main__.py         # Entry point for CLI execution
│   │   ├── cli.py              # Command-line interface
│   │   ├── model.py            # Loading models and running predictions
│   │   └── utils.py            # Helper functions and utilities
│   ├── checkpoints/            # Pretrained model weights
│   ├── examples/               # Example input and output files
│   ├── extract.py              # Script for ESM-2 embedding
│   ├── prott5_embedder.py      # Script for ProtT5 embedding
│   └── run_all.py              # Script to run full workflow
├── images/                     # Figures and diagrams used in notebooks or README
├── inference-app/              # Contains code for the web-based prediction tool hosted on Hugging Face Spaces.
├── requirements.txt            # Python dependencies
```
### Notebooks in detail:

- **feature-extraction**
  - [1. ESM-v2-embeddings.ipynb](notebooks/feature-extraction/1.%20ESM-v2-embeddings.ipynb): Extracts embeddings using [ESM-v2 model](https://github.com/facebookresearch/esm). Input protein sequences in FASTA format.
  - [2. ProtT5-embeddings.ipynb](notebooks/feature-extraction/2.%20ProtT5-embeddings.ipynb): Extracts embeddings using [ProtT5 model](https://github.com/agemagician/ProtTrans). Input protein sequences in FASTA format.
  - [3. AAC-feature-vectors.ipynb](notebooks/feature-extraction/3.%20AAC-feature-vectors.ipynb): Generates amino acid composition feature vectors. Input protein sequences in FASTA format.

- **modeling**
  - [classic-machine-learning.ipynb](notebooks/modeling/classic-machine-learning.ipynb): Classic machine learning models' training and evaluation, including SVM, RF, XGBoost, and KNN. This notebook also tests the effect of hyperparameter tuning and the autoencoder.
  - [nonlinear-DNN.ipynb](notebooks/modeling/nonlinear-DNN.ipynb): Train and evaluation of our top-performing deep neural network models, using ESM-v2 and ProtT5 embeddings, and AAC feature vectors. This notebook requires the pretrained model weights located in ```src/checkpoints/``` to run evaluations and reproduce results.
  - [single-layer-LSTM.ipynb](notebooks/modeling/single-layer-LSTM.ipynb): Training and evaluation of a single-layer LSTM (Long Short-Term Memory) model.
  - [1D-CNN.ipynb](notebooks/modeling/1D-CNN.ipynb): Training and evaluation of a 1-dimensional CNN (Convolutional neural network) model.
 
<h2 align="center">
  General AllerTrans Model Architecture
</h2>

![Model Architecture](images/Arch-AllerTrans.jpg)

## Dataset
The utilized dataset in this study is the public AlgPred 2.0 train and validation sets, that are available [here](https://webs.iiitd.edu.in/raghava/algpred2/stand.html).

---

## CLI Usage for Inference

### 1. Install Requirements

```bash
git clone https://github.com/faezesarlakifar/AllerTrans.git
cd AllerTrans
pip install -r requirements.txt
```

> Make sure torch CPU-only is fine.

### 2. Run Predictions

```bash
cd src
python run_all.py --fasta examples/protein_sequences.fasta --output examples/predictions.csv
```

* `--fasta`: Path to your input FASTA file (single or multi-sequence).
* `--output`: CSV file to save predictions.

---

#### Example Input

File: `examples/protein_sequences.fasta`

```
>Sequence_1
MQEAGAVKFDIKNQCGYTVWAAGLPGGGKRLDQGQTWTVNLAAGTASARFWGRTGCTFDASGKGSCQTGDCGRQLSCTVSGAVPATLAEYTQSDQDY
>Sequence_2
MSIQQIIEQKIQKEFQPHFLAIENESHLHHSNRGSESHFKCVIVSADFKNIRKVQRHQRIYQLLNEEL
...
```

#### Example Output

File: `examples/predictions.csv`

|     id      | prediction         |
| ----------- | ------------------ |
| Sequence_1  | Potential Allergen |
| Sequence_2  | Non-Allergen       |

Replace our `examples/protein_sequences.fasta` with your own FASTA file containing the sequences you want to classify.

---

## **Citation**

If our work contributes to your research, please cite:

```bibtex
@ARTICLE{AllerTrans2025,
  author  = {Sarlakifar, Faezeh and Malek, Hamed and Allahyari Fard, Najaf},
  title   = {AllerTrans: a deep learning method for predicting the allergenicity of protein sequences},
  journal = {Biology Methods and Protocols},
  year    = {2025},
  volume  = {10},
  number  = {1},
  doi     = {10.1093/biomethods/bpaf040}
}
```

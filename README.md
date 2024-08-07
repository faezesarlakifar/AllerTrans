<h1 align="center">
  AllerTrans
</h1>
<h2 align="center">
  An Improved Protein Allergenicity Prediction Using Deep Learning Model
</h2>

## Overview
AllerTrans is a deep learning model designed to predict the potential allergenicity of proteins based on their primary structure. We utilized two protein language models (ESM-v2 and ProtT5) to extract distinct feature vectors for each sequence. Our approach combines these vectors and inputs them into a deep neural network for classification. Our model classifies proteins into allergenic or non-allergenic categories, demonstrating admissible improvement in all reported evaluation metrics in the [AlgPred 2.0](https://doi.org/10.1093/bib/bbaa294). AllerTrans achieves a sensitivity of 97.91%, specificity of 97.69%, accuracy of 97.80%, and an impressive area under the ROC curve of 99% using standard five-fold cross-validation on the AlgPred 2.0 dataset.

<h2 align="center">
  A comprehensive flowchart that includes all of our experiments
</h2>

![Experiments' Flowchart](images/flowchart.jpg)

## Repository Structure

### Folders

- **feature-extraction**
  - [1. ESM-v2-embeddings.ipynb](feature-extraction/1.%20ESM-v2-embeddings.ipynb): Extracts embeddings using [ESM-v2 model](https://github.com/facebookresearch/esm). Input protein sequences in FASTA format.
  - [2. ProtT5-embeddings.ipynb](feature-extraction/2.%20ProtT5-embeddings.ipynb): Extracts embeddings using [ProtT5 model](https://github.com/agemagician/ProtTrans). Input protein sequences in FASTA format.
  - [3. AAC-feature-vectors.ipynb](feature-extraction/3.%20AAC-feature-vectors.ipynb): Generates amino acid composition feature vectors. Input protein sequences in FASTA format.

- **modeling**
  - [classic-machine-learning.ipynb](modeling/classic-machine-learning.ipynb): Classic machine learning models' training and evaluating, including SVM, RF, XGBoost, and KNN. This notebook also tests the effect of hyperparameter tuning and autoencoder.
  - [nonlinear-DNN.ipynb](modeling/nonlinear-DNN.ipynb): Train and evaluation of our top-performing deep neural network models, using ESM-v2 and ProtT5 embeddings, and AAC feature vectors.
  - [single-layer-LSTM.ipynb](modeling/single-layer-LSTM.ipynb): Training and evaluation of a single-layer LSTM (Long Short-Term Memory) model.
  - [1D-CNN.ipynb](modeling/1D-CNN.ipynb): Training and evaluation of a 1-dimensional CNN (Convolutional neural network) model.

- **model-checkpoints**
  - Contains saved checkpoints of the trained models required for the `nonlinear-DNN` notebook.

## Dataset
The utilized dataset for this study includes the public AlgPred 2.0 train and validation sets, which are available [here](https://webs.iiitd.edu.in/raghava/algpred2/stand.html).

## Usage

1. **Feature Extraction**:
   - Navigate to the `feature-extraction` folder and run the notebooks to extract the necessary feature vectors from protein sequences. Input protein sequences in FASTA format.

2. **Model Training and Evaluation**:
   - Navigate to the `modeling` folder.
   - Open and run the `nonlinear-DNN.ipynb` notebook to train and evaluate the deep neural network model. Ensure the required model checkpoints are available in the `model-checkpoints` folder.
   - For other models, run the respective notebooks (`classic-machine-learning.ipynb`, `single-layer-LSTM.ipynb`, `1D-CNN.ipynb`).

## Acknowledgements

We want to thank the developers of the ESM-v2 and ProtT5 models for providing the tools necessary for our feature extraction. Additionally, we acknowledge the public AlgPred 2.0 dataset, which helped us develop and evaluate our model.

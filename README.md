<h1 align="center">
  AllerTrans
</h1>
<h2 align="center">
  A Deep Learning Method for Predicting the Allergenicity of Protein Sequences
</h2>

## Overview
Recognizing the potential allergenicity of proteins is essential for ensuring their safety. Allergens are a major concern in determining protein safety, especially with the increasing use of recombinant proteins in new medical products. These proteins need careful allergenicity assessment to guarantee their safety. However, traditional laboratory testing for allergenicity is expensive and time-consuming. To address this challenge, bioinformatics offers efficient and cost-effective alternatives for predicting protein allergenicity. In this study, we developed an enhanced deep-learning model to predict the potential allergenicity of proteins based on their primary structure represented as protein sequences. In simple terms, this model classifies proteins into allergenic or non-allergenic classes. Our approach utilizes two protein language models to extract distinct feature vectors for each sequence, which are then input into a deep neural network model for classification. Each feature vector represents a specific aspect of the protein sequence, and combining them enhances the outcomes. Finally, we effectively combined the predictions of our top-performing models using ensemble modeling techniques. This could balance the model's sensitivity and specificity and improve the outcome. Our proposed model demonstrates admissible improvement compared to existing models, achieving a sensitivity of 97.91%, specificity of 97.69%, accuracy of 97.80%, and an impressive area under the ROC curve of 99% using the standard five-fold cross-validation.

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

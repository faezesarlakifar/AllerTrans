[![AllerTrans](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/spaces/sfaezella/AllerTrans)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Code Ocean](https://img.shields.io/badge/Code%20Ocean-Open%20Capsule-blue)](https://doi.org/10.24433/CO.1381053.v1)

[![AllerTrans](https://img.shields.io/badge/bioRxiv-10.1101/2024.08.09.607419v2-red)](https://www.biorxiv.org/content/10.1101/2024.08.09.607419v2)


<h1 align="center">
  AllerTrans
</h1>
<h2 align="center">
  A Deep Learning Method for Predicting the Allergenicity of Protein Sequences
</h2>

## 🔍 Overview
Allergens are a major concern in protein safety, especially with the growing use of recombinant proteins in medical products. Traditional allergenicity tests are costly and time-consuming, prompting the need for efficient bioinformatics solutions. In this study, we developed an enhanced deep learning model that classifies proteins as allergenic or non-allergenic based on their sequences. Our method extracts features using two protein language models and combines them in a deep neural network, followed by ensemble modeling to improve performance. The proposed model achieved strong results: 97.91% sensitivity, 97.69% specificity, 97.80% accuracy, and a 99% AUC using five-fold cross-validation.

bioRxiv DOI: [https://doi.org/10.1101/2024.08.09.607419](https://doi.org/10.1101/2024.08.09.607419)

## 🚀 Online Prediction Tool
You can try out the AllerTrans model directly available on Hugging Face Spaces:
[https://huggingface.co/spaces/sfaezella/AllerTrans](https://huggingface.co/spaces/sfaezella/AllerTrans)

<h2 align="center">
  A comprehensive flowchart that includes all of our experiments
</h2>

![Experiments' Flowchart](images/flowchart.jpg)

## 🛠️ Repository Structure

- **feature-extraction**
  - [1. ESM-v2-embeddings.ipynb](feature-extraction/1.%20ESM-v2-embeddings.ipynb): Extracts embeddings using [ESM-v2 model](https://github.com/facebookresearch/esm). Input protein sequences in FASTA format.
  - [2. ProtT5-embeddings.ipynb](feature-extraction/2.%20ProtT5-embeddings.ipynb): Extracts embeddings using [ProtT5 model](https://github.com/agemagician/ProtTrans). Input protein sequences in FASTA format.
  - [3. AAC-feature-vectors.ipynb](feature-extraction/3.%20AAC-feature-vectors.ipynb): Generates amino acid composition feature vectors. Input protein sequences in FASTA format.

- **modeling**
  - [classic-machine-learning.ipynb](modeling/classic-machine-learning.ipynb): Classic machine learning models' training and evaluation, including SVM, RF, XGBoost, and KNN. This notebook also tests the effect of hyperparameter tuning and the autoencoder.
  - [nonlinear-DNN.ipynb](modeling/nonlinear-DNN.ipynb): Train and evaluation of our top-performing deep neural network models, using ESM-v2 and ProtT5 embeddings, and AAC feature vectors.
  - [single-layer-LSTM.ipynb](modeling/single-layer-LSTM.ipynb): Training and evaluation of a single-layer LSTM (Long Short-Term Memory) model.
  - [1D-CNN.ipynb](modeling/1D-CNN.ipynb): Training and evaluation of a 1-dimensional CNN (Convolutional neural network) model.

- **model-checkpoints**
  - Contains saved checkpoints of the trained models required for the `nonlinear-DNN` notebook.

- **additional-experiments**
  - Includes supplementary experiments and analyses beyond the core modeling workflows.
    
- **inference-app**
  - Contains code for the web-based prediction tool hosted on Hugging Face Spaces.
 
<h2 align="center">
  General AllerTrans Model Architecture
</h2>

![Model Architecture](images/Arch-AllerTrans.jpg)

## 📂 Dataset
The utilized dataset in this study is the public AlgPred 2.0 train and validation sets, which are available [here](https://webs.iiitd.edu.in/raghava/algpred2/stand.html).

## ⚙ Usage

1. **Feature Extraction**:
    ```bash
    cd feature-extraction
    ```
   - Run the notebooks in the `feature-extraction` folder to extract the necessary feature vectors from protein sequences.
   - Input protein sequences must be in FASTA format.

3. **Model Training and Evaluation**:
   ```bash
    cd modeling
    ```
   - Open and run the `nonlinear-DNN.ipynb` notebook to train and evaluate the deep neural network model. Ensure the required model checkpoints are available in the `model-checkpoints` folder.
   - For other models, run the respective notebooks (`classic-machine-learning.ipynb`, `single-layer-LSTM.ipynb`, `1D-CNN.ipynb`).

<!-----
<p align="center">
  <img src="images/Arch-AllerTrans.jpg" alt="Model Architecture" width="800"/>
</p>
>

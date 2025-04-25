import torch
import gradio as gr
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
import esm
from inference import load_models, predict_ensemble
from transformers import AutoTokenizer, AutoModel
import spaces

# Load trained models
model_protT5, model_cat = load_models()

# Load ProtT5 model
tokenizer_t5 = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model_t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model_t5 = model_t5.eval()

# Load the tokenizer and model
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer_esm = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name)

def extract_prott5_embedding(sequence):
    sequence = sequence.replace(" ", "")
    seq = " ".join(list(sequence))
    ids = tokenizer_t5(seq, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model_t5(**ids).last_hidden_state
    return torch.mean(embedding, dim=1)


# Extract ESM2 embedding
def extract_esm_embedding(sequence):
    # Tokenize the sequence
    inputs = tokenizer_esm(sequence, return_tensors="pt", padding=True, truncation=True)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = esm_model(**inputs)
    
    # Extract the embeddings from the 33rd layer (ESM2 layer)
    token_representations = outputs.last_hidden_state  # This is the default layer
    return torch.mean(token_representations[0, 1:len(sequence)+1], dim=0).unsqueeze(0)

def estimate_duration(sequence):
    # Estimate duration based on sequence length
    base_time = 30  # Base time in seconds
    time_per_residue = 0.5  # Estimated time per residue
    estimated_time = base_time + len(sequence) * time_per_residue
    return min(int(estimated_time), 300)  # Cap at 300 seconds

@spaces.GPU(duration=120)
def classify(sequence):
    protT5_emb = extract_prott5_embedding(sequence)
    esm_emb = extract_esm_embedding(sequence)
    concat = torch.cat((esm_emb, protT5_emb), dim=1)
    pred = predict_ensemble(protT5_emb, concat, model_protT5, model_cat)
    return "Potential Allergen" if pred.item() == 1 else "Non-Allergen"

description_md = """
## 📌 **About AllerTrans – A Powerful Tool for Predicting the Allergenicity of Protein Sequences**

**🧬 Input Format – FASTA Sequences:** This tool accepts protein sequences in FASTA format.

**🧾 Output Explanation** – AllerTrans classifies your input sequence into one of the following categories:
###### **🟢 Non-Allergen:** The protein is unlikely to cause an allergic reaction and can be considered safe regarding allergenicity.
###### **🔴 Potential Allergen:** The protein has the potential to trigger an allergic response or exhibit cross-reactivity in some individuals.

**🔎 Caution & Disclaimer:**
###### Our model has demonstrated promising performance on the AlgPred 2.0 validation set, which includes a wide range of allergenic and non-allergenic sequences from diverse sources. AllerTrans is also capable of handling recombinant proteins, as supported by additional evaluation using a recombinant protein dataset from UniProt. However, **we advise caution when using this tool on all constructs and modifications of recombinant proteins**. The model's generalizability across various recombinant scenarios has yet to be fully explored.

###### 🚨 Remember, AllerTrans is designed as a reliable screening tool. However, for clinical or regulatory decisions, always confirm the prediction results through experimental validation.
""" 

demo = gr.Interface(fn=classify,
                    inputs=gr.Textbox(lines=3, placeholder="Enter protein sequence..."),
                    outputs=gr.Label(label="Prediction"),
                    description=description_md)

if __name__ == "__main__":
    demo.launch()
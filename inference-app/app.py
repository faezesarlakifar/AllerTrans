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
model_t5 = model_t5.eval().to("cuda")

# Load the tokenizer and model
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer_esm = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name).to("cuda")

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


# def classify(sequence):
#    protT5_emb = extract_prott5_embedding(sequence)
#    esm_emb = extract_esm_embedding(sequence)
#    concat = torch.cat((esm_emb, protT5_emb), dim=1)
#    pred = predict_ensemble(protT5_emb, concat, model_protT5, model_cat)
#    return "Potential Allergen" if pred.item() == 1 else "Non-Allergen"
    

@spaces.GPU(duration=120)
def classify(sequence):
    protT5_emb = extract_prott5_embedding(sequence)
    esm_emb = extract_esm_embedding(sequence)
    concat = torch.cat((esm_emb, protT5_emb), dim=1)
    pred = predict_ensemble(protT5_emb, concat, model_protT5, model_cat)
    return "Potential Allergen" if pred.item() == 1 else "Non-Allergen"


demo = gr.Interface(fn=classify,
                    inputs=gr.Textbox(lines=3, placeholder="Enter protein sequence..."),
                    outputs=gr.Label(label="Prediction"))

if __name__ == "__main__":
    demo.launch()

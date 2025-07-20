import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import configparser
from pathlib import Path

# --- THE FIX: The Adapter/Wrapper Class ---
# This simple module acts as an adapter. Its forward method has a signature
# that the ONNX JIT tracer can understand (it accepts a dictionary).
# It then correctly unpacks that dictionary into keyword arguments for the real model.
class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_dict):
        # The key is the double-asterisk (**), which unpacks the dictionary
        # into keyword arguments (e.g., input_ids=..., attention_mask=...).
        return self.model(**input_dict)

def convert():
    """
    Loads a SentenceTransformer model and exports it to the ONNX format.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    MODEL_NAME = config['NLP']['sentence_transformer_model']
    
    output_path = Path("model.onnx")
    if output_path.exists():
        print("ONNX model already exists. Skipping conversion.")
        return

    print(f"Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Wrapping model in ONNX-compatible adapter...")
    # We wrap our loaded model in the new adapter class.
    wrapped_model = OnnxWrapper(model)
    wrapped_model.eval() # Set the model to evaluation mode

    tokenizer = model[0].tokenizer

    print("Creating dummy input for ONNX export...")
    dummy_text = "This is a dummy sentence."
    
    tokenized_input = tokenizer(
        [dummy_text],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Convert the BatchEncoding to a standard Python dictionary.
    onnx_input = dict(tokenized_input)
    
    print(f"Exporting model to ONNX format at: {output_path}")

    # Now, we export the wrapped_model, not the original model.
    torch.onnx.export(
        wrapped_model,
        args=(onnx_input,), # The input remains a dictionary in a tuple
        f=output_path,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['sentence_embedding'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_len'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_len'},
            'token_type_ids': {0: 'batch_size', 1: 'sequence_len'},
            'sentence_embedding': {0: 'batch_size'}
        },
        opset_version=11
    )
    print(f"Model successfully converted to {output_path}")

if __name__ == "__main__":
    convert()

import torch
from sentence_transformers import SentenceTransformer
import configparser
from pathlib import Path

def convert():
    config = configparser.ConfigParser()
    config.read('config.ini')
    MODEL_NAME = config['NLP']['sentence_transformer_model']
    
    output_path = Path("model.onnx")
    if output_path.exists():
        print("ONNX model already exists. Skipping conversion.")
        return

    print(f"Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model[0].tokenizer

    print("Creating dummy input for ONNX export...")
    dummy_text = "This is a dummy sentence."
    dummy_input = tokenizer(
        [dummy_text],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    print(f"Exporting model to ONNX format at: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input,),
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

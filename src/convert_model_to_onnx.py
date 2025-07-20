import torch
from sentence_transformers import SentenceTransformer
import configparser
from pathlib import Path

def convert():
    """
    Loads a SentenceTransformer model and exports it to the ONNX format.
    This allows for a much smaller and faster runtime environment.
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
    tokenizer = model[0].tokenizer

    print("Creating dummy input for ONNX export...")
    dummy_text = "This is a dummy sentence."
    
    # This still produces the "fancy box" BatchEncoding object
    tokenized_input = tokenizer(
        [dummy_text],
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # --- THIS IS THE FIX ---
    # We convert the "fancy box" into a plain, standard Python dictionary.
    # This is the format the ONNX exporter's JIT tracer understands.
    onnx_input = dict(tokenized_input)
    
    print(f"Exporting model to ONNX format at: {output_path}")
    # The ONNX export needs the model, a tuple of args,
    # the output path, and names for the input/output nodes.
    torch.onnx.export(
        model,
        # We pass our plain dictionary as the only argument inside a tuple.
        args=(onnx_input,),
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

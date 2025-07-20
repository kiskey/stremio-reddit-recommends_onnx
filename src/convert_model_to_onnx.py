import torch
from sentence_transformers import SentenceTransformer
import configparser
from pathlib import Path

def convert():
    """
    Loads a SentenceTransformer model and exports it to the ONNX format.
    This is the robust, production-ready method.
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
    model.eval() # Set the model to evaluation mode

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
    
    # --- THIS IS THE FIX ---
    # The ONNX exporter's tracer works most reliably when given a flat tuple
    # of tensors as input, not a dictionary. The order must match the
    # model's internal forward() method. For BERT-like models, this is standard.
    args_tuple = (
        onnx_input['input_ids'],
        onnx_input['attention_mask'],
        onnx_input['token_type_ids']
    )
    
    print(f"Exporting model to ONNX format at: {output_path}")

    # We now export the original model directly, but pass the tensors as a tuple.
    torch.onnx.export(
        model,
        args=args_tuple,
        f=output_path,
        # The input_names list maps the positional arguments from our tuple
        # to the named inputs that the final ONNX model will expect.
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

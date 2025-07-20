# src/convert_model_to_onnx.py
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import configparser
from pathlib import Path

def convert():
    """
    Loads a Hugging Face model and exports it to the ONNX format
    using the Optimum library, which is the recommended approach.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    MODEL_NAME = config['NLP']['sentence_transformer_model']
    
    # The output will be a directory containing model.onnx and other config files
    output_path = Path("onnx_model")
    if output_path.exists():
        print(f"ONNX model directory '{output_path}' already exists. Skipping conversion.")
        return

    print(f"Loading base model and tokenizer for: {MODEL_NAME}")
    # We use ORTModelForFeatureExtraction from Optimum, which handles the export
    model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Saving ONNX model and tokenizer to directory: {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Model successfully converted and saved to {output_path}")

if __name__ == "__main__":
    convert()

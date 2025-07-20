from sentence_transformers import SentenceTransformer
import configparser
from pathlib import Path

def convert():
    """
    Loads a SentenceTransformer model and exports it to the ONNX format
    using the library's official, built-in utility function.
    This is the robust, correct, and documented method.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    MODEL_NAME = config['NLP']['sentence_transformer_model']
    
    output_path = Path("model.onnx")
    if output_path.exists():
        print(f"ONNX model at '{output_path}' already exists. Skipping conversion.")
        return

    print(f"Loading SentenceTransformer model: {MODEL_NAME}")
    # Load the model from the HuggingFace hub
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Exporting model to ONNX format at: {output_path}...")
    
    # Use the dedicated, built-in function to handle the export.
    # This abstracts away all the complex, error-prone details of torch.onnx.export.
    model.save_to_onnx(output_path)
    
    print(f"Model successfully converted to {output_path}")

if __name__ == "__main__":
    convert()

# Standard library imports
import sys
import os

# Data handling libraries
import numpy as np
import pandas as pd
from scipy.io import arff

# Add the path to semantic model implementations
sys.path.append('./semantic-dataset-creation')

# Import custom semantic feature extractors
from extractor import DeepAutoencoder, ConvolutionalAutoencoder, DeepBeliefNetwork

# Load input data from .csv or .arff files
def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()  # Get file extension
    if ext == '.csv':
        df = pd.read_csv(file_path)  # Load CSV file
        X = df.select_dtypes(include=[np.number]).values  # Extract numeric features
        y = None  # Target labels not used here
    elif ext == '.arff':
        data, meta = arff.loadarff(file_path)  # Load ARFF file
        df = pd.DataFrame(data)
        X = df.select_dtypes(include=[np.number]).values  # Extract numeric features
        y = None
    else:
        raise ValueError(f"Unsupported file extension: {ext}")  # Reject unsupported formats
    return X, y  # Return feature matrix and dummy target

# Save extracted features to disk as .npy file
def save_features(features, out_path):
    np.save(out_path, features)

# Main execution function
def main():
    # Ensure script is called with input file argument
    if len(sys.argv) < 2:
        print("Usage: python encode_semantic_features.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]  # Get input file path
    base_name = os.path.splitext(os.path.basename(input_file))[0]  # Extract base name for output files

    # Load and preprocess the dataset
    X, y = load_data(input_file)
    print(f"Loaded data from {input_file} with shape {X.shape}")

    # === Deep Autoencoder ===
    print("Extracting features with DeepAutoencoder...")
    da_features = DeepAutoencoder().get_features(X, y)  # Extract semantic features
    save_features(da_features, f"{base_name}_DA_features.npy")  # Save to file
    print(f"Saved DA features to {base_name}_DA_features.npy")

    # === Convolutional Autoencoder ===
    print("Extracting features with ConvolutionalAutoencoder...")
    ca_features = ConvolutionalAutoencoder().get_features(X, y)
    save_features(ca_features, f"{base_name}_CA_features.npy")
    print(f"Saved CA features to {base_name}_CA_features.npy")

    # === Deep Belief Network ===
    print("Extracting features with DeepBeliefNetwork...")
    dbn_features = DeepBeliefNetwork().get_features(X, y)
    save_features(dbn_features, f"{base_name}_DBN_features.npy")
    print(f"Saved DBN features to {base_name}_DBN_features.npy")

# Run the main function when executed as a script
if __name__ == "__main__":
    main()

import sys
import os
import numpy as np
import pandas as pd
from scipy.io import arff
sys.path.append('./semantic-dataset-creation')
from extractor import DeepAutoencoder, ConvolutionalAutoencoder, DeepBeliefNetwork
 
def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(file_path)
        X = df.select_dtypes(include=[np.number]).values
        y = None
    elif ext == '.arff':
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        X = df.select_dtypes(include=[np.number]).values
        y = None
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return X, y
 
def save_features(features, out_path):
    np.save(out_path, features)
 
def main():
    if len(sys.argv) < 2:
        print("Usage: python encode_semantic_features.py <input_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    X, y = load_data(input_file)
 
    print(f"Loaded data from {input_file} with shape {X.shape}")
 
    # Deep Autoencoder
    print("Extracting features with DeepAutoencoder...")
    da_features = DeepAutoencoder().get_features(X, y)
    save_features(da_features, f"{base_name}_DA_features.npy")
    print(f"Saved DA features to {base_name}_DA_features.npy")
 
    # Convolutional Autoencoder
    print("Extracting features with ConvolutionalAutoencoder...")
    ca_features = ConvolutionalAutoencoder().get_features(X, y)
    save_features(ca_features, f"{base_name}_CA_features.npy")
    print(f"Saved CA features to {base_name}_CA_features.npy")
 
    # Deep Belief Network
    print("Extracting features with DeepBeliefNetwork...")
    dbn_features = DeepBeliefNetwork().get_features(X, y)
    save_features(dbn_features, f"{base_name}_DBN_features.npy")
    print(f"Saved DBN features to {base_name}_DBN_features.npy")
 
if __name__ == "__main__":
    main() 

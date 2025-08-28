#!/usr/bin/env python3
"""
REPD Semantic Comparison

- Trains a REPD model using the author's implementation (autoencoder.py + REPD_Impl.py)
  on the provided semantic dataset (openj9_metrics_DA_features.npy + data/openj9_metrics.csv).
- Evaluates provided Java files' semantic features (DA) for base/head states.
- Saves predictions to CSVs that match the workflow's expected format.

Usage:
  python repd_semantic_compare.py --files file1.java file2.java --output analysis_dir --state base

Multiple invocations are used by the workflow for base and head.
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure local imports work
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from REPD_Impl import REPD
from autoencoder import AutoEncoder
from java_feature_extractor import JavaFeatureExtractor

DATA_CSV = SCRIPT_DIR / "data" / "openj9_metrics.csv"
DA_FEATURES_NPY = SCRIPT_DIR / "openj9_metrics_DA_features.npy"


def load_training_data():
    if not DA_FEATURES_NPY.exists() or not DATA_CSV.exists():
        raise FileNotFoundError("Required training data not found: openj9_metrics_DA_features.npy and data/openj9_metrics.csv")
    X = np.load(str(DA_FEATURES_NPY))
    labels_df = pd.read_csv(DATA_CSV)
    if 'bug' not in labels_df.columns:
        raise ValueError("Column 'bug' not found in data/openj9_metrics.csv")
    y = (labels_df['bug'] > 0).astype(int).to_numpy()
    # If DBN-like shape, flatten last dim
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], X.shape[2])
    return X, y


def train_repd_author_impl() -> REPD:
    X, y = load_training_data()
    input_dim = X.shape[1]
    # Simple architecture: input -> input/2 -> input/4 -> input/2 -> input
    # Match author's AutoEncoder API
    layers = [input_dim, max(2, input_dim // 2), max(2, input_dim // 4), max(2, input_dim // 2), input_dim]
    ae = AutoEncoder(layers=layers, lr=0.001, epoch=50, batch_size=256)
    repd = REPD(dim_reduction_model=ae)
    repd.fit(X, y)
    return repd


def evaluate_files(repd: REPD, java_files, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    extractor = JavaFeatureExtractor()
    # Extract DA features using our extractor (returns dict)
    features = extractor.extract_features_from_files(java_files)
    if 'DA' not in features or features['DA'] is None or len(features['DA']) == 0:
        # Write error
        with open(os.path.join(output_dir, 'error.txt'), 'w') as f:
            f.write('No semantic features extracted for provided files')
        return None
    X = features['DA']
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], X.shape[2])

    # Reconstruction errors and probabilities using REPD implementation
    errors = repd.calculate_reconstruction_error(X)
    p_nd = repd.get_non_defect_probability(errors)
    p_d = repd.get_defect_probability(errors)
    preds = (p_d > p_nd).astype(int)

    rows = []
    for i, fpath in enumerate(java_files):
        prob_def = float(p_d[i]) if np.ndim(p_d) else float(p_d)
        rows.append({
            'file_path': fpath,
            'file_id': os.path.basename(fpath),
            'prediction': int(preds[i]),
            'probability_defective': prob_def,
            'reconstruction_error': float(errors[i]) if np.ndim(errors) else float(errors),
            'model_type': 'DA'
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, 'files_analysis_DA_predictions.csv')
    df.to_csv(out_csv, index=False)

    # Save a simple summary JSON
    summary = {
        'total_files': len(rows),
        'defective_predicted': int(df['prediction'].sum()),
        'avg_prob_defective': float(df['probability_defective'].mean()),
        'avg_reconstruction_error': float(df['reconstruction_error'].mean())
    }
    with open(os.path.join(output_dir, 'files_analysis_prediction_summary.json'), 'w') as f:
        json.dump(summary, f)
    return df


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True, help='Java files to analyze')
    ap.add_argument('--output', required=True, help='Output directory')
    ap.add_argument('--state', choices=['base', 'head'], default='head')
    return ap.parse_args()


def main():
    args = parse_args()
    repd = train_repd_author_impl()
    evaluate_files(repd, args.files, args.output)


if __name__ == '__main__':
    main() 
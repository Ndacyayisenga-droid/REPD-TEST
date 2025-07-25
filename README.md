# 📌 Summary Checklist for You

| Task                                               | Status      |
| -------------------------------------------------- | ----------- |
| Semantic data collected                            | ✅ Done      |
| Encode semantic features (DBN, DA, CA)             | ✅ Done  |
| Train REPD model on non-defective examples         | 🔜 Do next  |
| Compute reconstruction error and fit distributions | 🔜 Do next  |
| Classify unseen data with REPD                     | 🔜 Do next  |
| Compare performance with baseline models           | 🔜 Optional |
| Test robustness to class imbalance                 | 🔜 Optional |

# Semantic data collected (Running the Metrics Script)

To generate a dataset of software metrics for a Java project using the provided script, follow these steps:

---

## Prerequisites:

- Ensure git and Python 3 are installed on your system.
- The script requires the javalang Python library, which will be installed automatically in a virtual environment.

## Steps:

1. Run the script with a GitHub repository URL as an argument:
   ```bash
   bash analyze_metrics.sh https://github.com/eclipse-openj9/openj9
   ```

The script will:
- Clone the repository.
- Set up a Python virtual environment and install javalang.
- Analyze all `.java` files to compute metrics (e.g., WMC, RFC, LOC, CBO, CA, CE, etc.).
- Generate a CSV file in the `metrics_output` directory named `<repository_name>_metrics.csv`.

## Output:

- The CSV file contains metrics for each Java class, including complexity, coupling, cohesion, inheritance, and bug counts derived from Git commit messages.
- Check the console output for success or error messages.

## Notes:

- Ensure a stable internet connection for cloning the repository.
- The script cleans up temporary files after execution.
- For accurate metrics like ca, ce, noc, mfa, cbm, and cam, the script performs a global analysis of class dependencies and inheritance.

---

# Encode semantic features (DBN, DA, CA)

This repository provides tools to extract semantic features from datasets using three types of neural models:
- **Deep Autoencoder (DA)**
- **Convolutional Autoencoder (CA)**
- **Deep Belief Network (DBN)**

The main script for feature extraction is `encode_semantic_features.py`.

## Prerequisites

- Python 3.8+
- Install required packages:
  - numpy
  - pandas
  - scipy
  - keras
  - torch

You can install the dependencies with:
```bash
pip install numpy pandas scipy keras torch
```

## Data Format

- Input files can be in `.csv` or `.arff` format.
- The script expects the input file to contain only numeric features (non-numeric columns are ignored).
- Labels are optional. If present, they should be in a separate column (not required for feature extraction).

## Usage Example

To encode semantic features from a dataset (e.g., `data/openj9_metrics.csv`):

```bash
python encode_semantic_features.py data/openj9_metrics.csv
```

This will generate three files in the current directory:
- `openj9_metrics_DA_features.npy`  (Deep Autoencoder features)
- `openj9_metrics_CA_features.npy`  (Convolutional Autoencoder features)
- `openj9_metrics_DBN_features.npy` (Deep Belief Network features)

## How it Works

- The script loads the input data and applies each model in turn:
  1. **Deep Autoencoder**: Extracts compressed representations.
  2. **Convolutional Autoencoder**: Extracts features using convolutional layers.
  3. **Deep Belief Network**: Extracts features using a stack of Restricted Boltzmann Machines.
- Each model's features are saved as a `.npy` file with a suffix indicating the model.

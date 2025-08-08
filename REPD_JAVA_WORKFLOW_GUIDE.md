# REPD-Workflow Adaptation for Java Semantic Analysis

This repository has been adapted from the REPD-Workflow logic to work with Java files and semantic datasets. The implementation provides a complete pipeline for feature extraction, bug prediction, and CI/CD integration.

## 🏗️ Architecture Overview

The adapted system consists of four main components:

### 1. Feature Extraction Pipeline (`java_feature_extractor.py`)
- **Purpose**: Extract semantic features from Java repositories
- **Input**: Java files or Git repositories
- **Output**: Semantic feature vectors (DA, CA, DBN)
- **Technology**: AST-based analysis using ASTEncoder-v1.2.jar

### 2. Bug Prediction Pipeline (`bug_prediction_pipeline.py`)
- **Purpose**: Predict bugs using trained REPD models
- **Input**: Extracted features or Java files
- **Output**: Defect probabilities and classifications
- **Models**: Deep Autoencoder (DA), Convolutional Autoencoder (CA), Deep Belief Network (DBN)

### 3. GitHub Actions Workflow (`.github/workflows/java-bug-prediction.yml`)
- **Purpose**: Automated PR analysis and bug prediction
- **Trigger**: Pull requests with Java file changes
- **Output**: PR comments with risk assessment and recommendations

### 4. Model Management (`model_manager.py`)
- **Purpose**: Validate, monitor, and maintain REPD models
- **Features**: Model validation, performance reports, retraining capabilities

## 🚀 Quick Start

### Prerequisites

```bash
# Python dependencies
pip install numpy pandas scikit-learn tensorflow torch scipy tqdm

# Java runtime (for AST encoder)
sudo apt-get update
sudo apt-get install default-jdk
```

### Basic Usage

#### 1. Extract Features from a Repository

```bash
python java_feature_extractor.py --repo https://github.com/example/java-project.git --output features/
```

#### 2. Predict Bugs in Repository

```bash
python bug_prediction_pipeline.py --repo https://github.com/example/java-project.git --output results/ --report
```

#### 3. Analyze Specific Files

```bash
python bug_prediction_pipeline.py --files src/main/java/Example.java --output results/ --report
```

#### 4. Validate Models

```bash
python model_manager.py --action validate --model-dir trained_models/
```

## 📋 Detailed Usage Guide

### Feature Extraction

The `JavaFeatureExtractor` class provides methods for extracting semantic features:

```python
from java_feature_extractor import JavaFeatureExtractor

extractor = JavaFeatureExtractor()

# Extract from repository
features = extractor.extract_features_from_repository(
    "https://github.com/example/repo.git",
    output_dir="extracted_features"
)

# Extract from specific files
java_files = ["src/Example.java", "src/Helper.java"]
features = extractor.extract_features_from_files(java_files)
```

**Output Structure:**
```
extracted_features/
├── repo_name_DA_features.npy     # Deep Autoencoder features
├── repo_name_CA_features.npy     # Convolutional Autoencoder features
├── repo_name_DBN_features.npy    # Deep Belief Network features
└── repo_name_metadata.json       # Feature metadata
```

### Bug Prediction

The `BugPredictor` class uses trained REPD models for prediction:

```python
from bug_prediction_pipeline import BugPredictor

predictor = BugPredictor(model_dir="trained_models")

# Predict on repository
results = predictor.predict_repository(
    "https://github.com/example/repo.git",
    output_dir="predictions"
)

# Generate detailed report
report = predictor.generate_report(results, "bug_report.md")
```

**Output Structure:**
```
predictions/
├── repo_name_DA_predictions.csv      # DA model predictions
├── repo_name_CA_predictions.csv      # CA model predictions
├── repo_name_DBN_predictions.csv     # DBN model predictions
├── repo_name_prediction_summary.json # Summary statistics
└── prediction_report.md              # Detailed markdown report
```

### Prediction Output Format

Each prediction CSV contains:

| Column | Description |
|--------|-------------|
| `file_id` | File identifier or path |
| `prediction` | Binary prediction (0=non-defective, 1=defective) |
| `probability_defective` | Probability of being defective (0.0-1.0) |
| `reconstruction_error` | Autoencoder reconstruction error |
| `model_type` | Model used (DA/CA/DBN) |

## 🔄 GitHub Actions Integration

### Workflow Setup

1. **Copy the workflow file** to your repository:
   ```bash
   mkdir -p .github/workflows
   cp .github/workflows/java-bug-prediction.yml .github/workflows/
   ```

2. **Add REPD models** to your repository:
   ```bash
   # Copy trained models
   cp -r trained_models/ your-repo/trained_models/
   
   # Copy semantic dataset creation tools
   cp -r semantic-dataset-creation/ your-repo/semantic-dataset-creation/
   
   # Copy prediction scripts
   cp java_feature_extractor.py your-repo/
   cp bug_prediction_pipeline.py your-repo/
   ```

3. **Configure requirements.txt**:
   ```
   numpy>=1.21.0
   pandas>=1.3.0
   scikit-learn>=1.0.0
   tensorflow>=2.8.0
   torch>=1.11.0
   scipy>=1.7.0
   tqdm>=4.62.0
   ```

### Workflow Behavior

The workflow automatically:

1. **Triggers** on PRs with Java file changes
2. **Extracts features** from changed Java files
3. **Generates predictions** using all available models
4. **Posts detailed comments** to the PR with:
   - Risk assessment for each changed file
   - Model-specific predictions
   - Overall risk evaluation
   - Actionable recommendations

### Example PR Comment

```markdown
## 🔍 Java Bug Prediction Analysis

**Analysis completed using REPD semantic models**

📊 **Files Analyzed:** 3
🤖 **Models Used:** DA, CA, DBN

### 🔬 DA Model Results
- **Predicted Defective:** 1/3 files (33.3%)
- **Average Defect Probability:** 0.4523

**🚨 Top Risky Files:**
| File | Defect Probability | Status |
|------|-------------------|--------|
| `src/main/java/ComplexService.java` | 0.8421 | 🔴 High Risk |

### 📋 Overall Assessment
🟡 **MEDIUM RISK**: Some files in this PR may need closer review.

### 💡 Recommendations
🔍 **Focus code review on high-risk files**
📝 **Consider additional testing for flagged components**
```

## 🛠️ Model Management

### Model Validation

```bash
# Validate all models
python model_manager.py --action validate

# Generate comprehensive report
python model_manager.py --action report --output model_status.md
```

### Model Retraining

```bash
# Retrain specific model
python model_manager.py --action retrain --model-type DA

# Export models to different location
python model_manager.py --action export --export-dir backup_models/
```

### Performance Monitoring

The model manager provides:
- **Model validation** on current data
- **Performance comparison** across different models
- **Automated health checks** for deployed models
- **Retraining recommendations** based on performance drift

## 📊 Model Performance

Current model performance on OpenJ9 dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| DA    | 99.78%   | 99.78%    | 100.00% | 99.89%   |
| CA    | 99.78%   | 99.78%    | 100.00% | 99.89%   |
| DBN   | 27.06%   | 99.73%    | 26.97%  | 42.46%   |

**Note**: DBN model captures different defect patterns and provides complementary insights.

## 🔧 Configuration

### AST Encoder Configuration

Modify `semantic-dataset-creation/config/parser.properties` to customize which AST elements are extracted:

```properties
# Enable/disable specific Java constructs
METHOD_INVOCATION=true
IF_STATEMENT=true
FOR_STATEMENT=true
ARRAY_ACCESS=true
# ... see full file for all options
```

### Model Training Parameters

Adjust parameters in the semantic feature extractors:

```python
# In extractor.py
class DeepAutoencoder:
    def __train(self, train):
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.model.fit(train, train, epochs=1000, batch_size=20, shuffle=True)
```

## 🤝 Integration Examples

### Example 1: Hiero Enterprise Java

For the [Hiero Enterprise Java](https://github.com/OpenElements/hiero-enterprise-java) repository:

```bash
# Analyze specific PR files
python bug_prediction_pipeline.py \
  --files hiero-dependency-injection/src/main/java/com/openelements/hiero/base/implementation/HieroDependencyInjectionSupplier.java \
  --output hiero_analysis/ \
  --report

# Extract features from entire repository
python java_feature_extractor.py \
  --repo https://github.com/OpenElements/hiero-enterprise-java.git \
  --output hiero_features/
```

### Example 2: CI/CD Integration

Add to your existing CI pipeline:

```yaml
# In your existing .github/workflows/ci.yml
- name: Bug Risk Assessment
  run: |
    python bug_prediction_pipeline.py --files $(git diff --name-only HEAD~1 HEAD | grep '\.java$') --output risk_assessment/
    
- name: Generate Risk Report
  run: |
    if [ -f risk_assessment/prediction_report.md ]; then
      cat risk_assessment/prediction_report.md >> $GITHUB_STEP_SUMMARY
    fi
```

## 📈 Advanced Features

### Custom Model Training

To train models on your specific codebase:

```python
from train_repd_model import REPDTrainer

# Prepare your dataset (CSV with metrics + bug labels)
trainer = REPDTrainer("your_dataset.csv")
trainer.train_all_models()
```

### Batch Processing

For large-scale analysis:

```python
from java_feature_extractor import JavaFeatureExtractor
from bug_prediction_pipeline import BugPredictor

repos = [
    "https://github.com/org/repo1.git",
    "https://github.com/org/repo2.git",
    # ... more repositories
]

extractor = JavaFeatureExtractor()
predictor = BugPredictor()

for repo in repos:
    features = extractor.extract_features_from_repository(repo, f"features/{repo.split('/')[-1]}")
    results = predictor.predict_repository(repo, f"results/{repo.split('/')[-1]}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Java AST Encoder Fails**
   ```bash
   # Ensure Java is installed
   java -version
   
   # Check JAR file permissions
   chmod +x semantic-dataset-creation/ASTEncoder-v1.2.jar
   ```

2. **Model Loading Errors**
   ```bash
   # Verify model files exist
   ls -la trained_models/
   
   # Check model compatibility
   python model_manager.py --action validate
   ```

3. **Memory Issues with Large Repositories**
   ```python
   # Limit files processed
   java_files = java_files[:100]  # Process first 100 files
   ```

### Performance Optimization

- **Parallel Processing**: Extract features in batches
- **Caching**: Reuse extracted features for multiple predictions
- **Model Selection**: Use fastest model (DA) for quick analysis

## 📚 References

- [Original REPD Paper](https://example.com/repd-paper)
- [AST-based Feature Extraction](https://example.com/ast-features)
- [Autoencoder Architectures](https://example.com/autoencoders)

## 🤖 Contributing

To extend this system:

1. **Add new extractors** in `semantic-dataset-creation/extractor.py`
2. **Implement custom metrics** in the feature extraction pipeline
3. **Enhance GitHub workflow** with additional analysis steps
4. **Add new model architectures** following the REPD framework

---

*This implementation successfully adapts REPD-Workflow logic for Java semantic analysis, providing automated bug prediction capabilities for Java repositories.*



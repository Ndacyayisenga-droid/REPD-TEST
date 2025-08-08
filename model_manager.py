#!/usr/bin/env python3
"""
Model Management Utilities

This module provides utilities for managing REPD models, including
model validation, performance monitoring, and retraining capabilities.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Utility class for managing REPD models and monitoring their performance.
    """
    
    def __init__(self, model_dir: str = "trained_models"):
        """
        Initialize the model manager.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.training_history = {}
        
    def load_models(self) -> None:
        """Load all available trained models."""
        logger.info(f"Loading models from {self.model_dir}")
        
        # Load training results
        results_path = os.path.join(self.model_dir, "training_results.pkl")
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                training_results = pickle.load(f)
                self.scalers = training_results.get('scalers', {})
                self.training_history = training_results.get('results', {})
        
        # Load individual models
        model_types = ['DA', 'CA', 'DBN']
        for model_type in model_types:
            model_path = os.path.join(self.model_dir, f"repd_model_{model_type}.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_type] = pickle.load(f)
                logger.info(f"Loaded {model_type} model")
    
    def validate_models(self, test_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict]:
        """
        Validate loaded models on test data.
        
        Args:
            test_data: Optional test data dictionary with feature arrays
            
        Returns:
            Validation results for each model
        """
        logger.info("Validating models")
        
        if not self.models:
            self.load_models()
        
        validation_results = {}
        
        for model_type, model in self.models.items():
            try:
                # Use stored training history if no test data provided
                if test_data is None or model_type not in test_data:
                    if model_type in self.training_history:
                        validation_results[model_type] = self.training_history[model_type]
                    else:
                        validation_results[model_type] = {"status": "No validation data available"}
                    continue
                
                # Validate on provided test data
                features = test_data[model_type]
                scaler = self.scalers.get(model_type)
                
                if scaler:
                    features = scaler.transform(features)
                
                # Generate predictions
                predictions = model.predict(features)
                probabilities = model.predict_proba(features)
                
                validation_results[model_type] = {
                    "status": "validated",
                    "predictions_shape": predictions.shape,
                    "probabilities_shape": probabilities.shape,
                    "prediction_distribution": {
                        "defective": int(np.sum(predictions == 1)),
                        "non_defective": int(np.sum(predictions == 0))
                    }
                }
                
            except Exception as e:
                logger.error(f"Validation failed for {model_type}: {e}")
                validation_results[model_type] = {"status": "validation_failed", "error": str(e)}
        
        return validation_results
    
    def generate_model_report(self, output_file: str = "model_report.md") -> str:
        """
        Generate a comprehensive model report.
        
        Args:
            output_file: Output markdown file path
            
        Returns:
            Report content as string
        """
        if not self.models:
            self.load_models()
        
        report_lines = [
            "# REPD Model Report",
            "",
            f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Models Directory:** {self.model_dir}",
            "",
            "## Available Models",
            ""
        ]
        
        for model_type in self.models.keys():
            model_path = os.path.join(self.model_dir, f"repd_model_{model_type}.pkl")
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            report_lines.extend([
                f"### {model_type} Model",
                "",
                f"- **File:** `repd_model_{model_type}.pkl`",
                f"- **Size:** {file_size:.2f} MB",
                f"- **Status:** ✅ Loaded"
            ])
            
            # Add training history if available
            if model_type in self.training_history:
                history = self.training_history[model_type]
                if isinstance(history, dict):
                    report_lines.extend([
                        f"- **Accuracy:** {history.get('accuracy', 'N/A')}",
                        f"- **Precision:** {history.get('precision', 'N/A')}",
                        f"- **Recall:** {history.get('recall', 'N/A')}",
                        f"- **F1-Score:** {history.get('f1_score', 'N/A')}"
                    ])
            
            report_lines.append("")
        
        # Model validation
        validation_results = self.validate_models()
        
        report_lines.extend([
            "## Model Validation",
            ""
        ])
        
        for model_type, results in validation_results.items():
            status_emoji = "✅" if results.get("status") == "validated" else "⚠️"
            report_lines.extend([
                f"### {model_type} Validation {status_emoji}",
                ""
            ])
            
            if "prediction_distribution" in results:
                dist = results["prediction_distribution"]
                total = dist["defective"] + dist["non_defective"]
                defect_rate = (dist["defective"] / total * 100) if total > 0 else 0
                
                report_lines.extend([
                    f"- **Prediction Distribution:**",
                    f"  - Defective: {dist['defective']} ({defect_rate:.1f}%)",
                    f"  - Non-defective: {dist['non_defective']} ({100-defect_rate:.1f}%)"
                ])
            
            if "error" in results:
                report_lines.append(f"- **Error:** {results['error']}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        working_models = [k for k, v in validation_results.items() if v.get("status") == "validated"]
        if len(working_models) == len(self.models):
            report_lines.append("✅ All models are working correctly")
        elif working_models:
            report_lines.append(f"⚠️ {len(working_models)}/{len(self.models)} models are working")
            failed_models = [k for k in self.models.keys() if k not in working_models]
            report_lines.append(f"Failed models: {', '.join(failed_models)}")
        else:
            report_lines.append("🚨 No models are working correctly - retrain required")
        
        report_lines.extend([
            "",
            "---",
            "*Generated by REPD Model Manager*"
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Model report saved to {output_file}")
        return report_content
    
    def retrain_model(self, model_type: str, data_path: str = "data/openj9_metrics.csv") -> Dict:
        """
        Retrain a specific model with new data.
        
        Args:
            model_type: Type of model to retrain (DA, CA, DBN)
            data_path: Path to training data
            
        Returns:
            Retraining results
        """
        logger.info(f"Retraining {model_type} model")
        
        try:
            # Import training modules
            from train_repd_model import REPDTrainer
            
            # Initialize trainer
            trainer = REPDTrainer(data_path)
            
            # Train specific model
            results = trainer.train_repd_model(model_type)
            
            logger.info(f"Successfully retrained {model_type} model")
            return {"status": "success", "results": results}
            
        except Exception as e:
            logger.error(f"Failed to retrain {model_type} model: {e}")
            return {"status": "failed", "error": str(e)}
    
    def compare_models(self, test_features: Dict[str, np.ndarray], true_labels: np.ndarray) -> pd.DataFrame:
        """
        Compare performance of different models on the same test data.
        
        Args:
            test_features: Dictionary of test features for each model type
            true_labels: True labels for comparison
            
        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing model performance")
        
        if not self.models:
            self.load_models()
        
        comparison_results = []
        
        for model_type, model in self.models.items():
            if model_type not in test_features:
                continue
                
            try:
                features = test_features[model_type]
                scaler = self.scalers.get(model_type)
                
                if scaler:
                    features = scaler.transform(features)
                
                predictions = model.predict(features)
                probabilities = model.predict_proba(features)
                
                # Calculate metrics
                accuracy = accuracy_score(true_labels, predictions)
                precision = precision_score(true_labels, predictions, average='weighted')
                recall = recall_score(true_labels, predictions, average='weighted')
                f1 = f1_score(true_labels, predictions, average='weighted')
                
                comparison_results.append({
                    'Model': model_type,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Test_Samples': len(features)
                })
                
            except Exception as e:
                logger.error(f"Comparison failed for {model_type}: {e}")
                comparison_results.append({
                    'Model': model_type,
                    'Accuracy': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'F1-Score': np.nan,
                    'Test_Samples': 0,
                    'Error': str(e)
                })
        
        return pd.DataFrame(comparison_results)
    
    def export_models(self, export_dir: str) -> None:
        """
        Export models to a different directory.
        
        Args:
            export_dir: Directory to export models to
        """
        logger.info(f"Exporting models to {export_dir}")
        
        os.makedirs(export_dir, exist_ok=True)
        
        # Copy model files
        import shutil
        for model_type in self.models.keys():
            src_path = os.path.join(self.model_dir, f"repd_model_{model_type}.pkl")
            dst_path = os.path.join(export_dir, f"repd_model_{model_type}.pkl")
            shutil.copy2(src_path, dst_path)
        
        # Copy training results
        src_results = os.path.join(self.model_dir, "training_results.pkl")
        if os.path.exists(src_results):
            dst_results = os.path.join(export_dir, "training_results.pkl")
            shutil.copy2(src_results, dst_results)
        
        logger.info(f"Models exported to {export_dir}")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage REPD models")
    parser.add_argument("--action", choices=["validate", "report", "retrain", "export"], 
                       required=True, help="Action to perform")
    parser.add_argument("--model-dir", default="trained_models", help="Model directory")
    parser.add_argument("--model-type", help="Model type for retraining")
    parser.add_argument("--export-dir", help="Export directory")
    parser.add_argument("--output", help="Output file for reports")
    
    args = parser.parse_args()
    
    manager = ModelManager(args.model_dir)
    
    if args.action == "validate":
        results = manager.validate_models()
        print("Validation Results:")
        for model_type, result in results.items():
            print(f"  {model_type}: {result.get('status', 'unknown')}")
    
    elif args.action == "report":
        output_file = args.output or "model_report.md"
        report = manager.generate_model_report(output_file)
        print(f"Report generated: {output_file}")
    
    elif args.action == "retrain":
        if not args.model_type:
            parser.error("--model-type required for retraining")
        result = manager.retrain_model(args.model_type)
        print(f"Retraining result: {result['status']}")
    
    elif args.action == "export":
        if not args.export_dir:
            parser.error("--export-dir required for export")
        manager.export_models(args.export_dir)
        print(f"Models exported to {args.export_dir}")


if __name__ == "__main__":
    main()

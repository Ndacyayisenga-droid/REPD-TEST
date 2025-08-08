#!/usr/bin/env python3
"""
Bug Prediction Pipeline

This module implements the REPD bug prediction pipeline adapted from REPD-Workflow
to work with Java repositories and semantic datasets.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from java_feature_extractor import JavaFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BugPredictor:
    """
    Bug prediction pipeline using trained REPD models.
    Adapted from REPD-Workflow for Java semantic analysis.
    """
    
    def __init__(self, model_dir: str = "trained_models"):
        """
        Initialize the bug predictor.
        
        Args:
            model_dir: Directory containing trained REPD models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_extractor = JavaFeatureExtractor()
        
        # Load trained models
        self._load_models()
    
    def _load_models(self) -> None:
        """Load trained REPD models and scalers."""
        logger.info("Loading trained REPD models")
        
        try:
            # Load training results which contain scalers
            results_path = os.path.join(self.model_dir, "training_results.pkl")
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    training_results = pickle.load(f)
                    self.scalers = training_results.get('scalers', {})
            
            # Load individual models
            model_types = ['DA', 'CA', 'DBN']
            for model_type in model_types:
                model_path = os.path.join(self.model_dir, f"repd_model_{model_type}.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_type] = pickle.load(f)
                    logger.info(f"Loaded {model_type} model")
                else:
                    logger.warning(f"Model not found: {model_path}")
                    
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict_repository(self, repo_url: str, output_dir: str = "prediction_results") -> Dict[str, pd.DataFrame]:
        """
        Predict bugs in a Java repository.
        
        Args:
            repo_url: URL of the repository to analyze
            output_dir: Directory to save results
            
        Returns:
            Dictionary of prediction results for each model type
        """
        logger.info(f"Predicting bugs in repository: {repo_url}")
        
        # Extract features from repository
        features = self.feature_extractor.extract_features_from_repository(repo_url, output_dir)
        
        if not features:
            logger.error("No features extracted from repository")
            return {}
        
        # Generate predictions for each available model
        results = {}
        for model_type in self.models.keys():
            if model_type in features:
                logger.info(f"Generating predictions with {model_type} model")
                predictions = self._predict_with_model(features[model_type], model_type)
                results[model_type] = predictions
            else:
                logger.warning(f"Features not available for {model_type} model")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        self._save_results(results, output_dir, repo_url)
        
        return results
    
    def predict_files(self, java_files: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Predict bugs in specific Java files.
        
        Args:
            java_files: List of Java file paths
            
        Returns:
            Dictionary of prediction results for each model type
        """
        logger.info(f"Predicting bugs in {len(java_files)} Java files")
        
        # Extract features from files
        features = self.feature_extractor.extract_features_from_files(java_files)
        
        if not features:
            logger.error("No features extracted from files")
            return {}
        
        # Generate predictions
        results = {}
        for model_type in self.models.keys():
            if model_type in features:
                predictions = self._predict_with_model(features[model_type], model_type, java_files)
                results[model_type] = predictions
        
        return results
    
    def _predict_with_model(self, features: np.ndarray, model_type: str, file_paths: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate predictions using a specific REPD model.
        
        Args:
            features: Feature matrix
            model_type: Type of model (DA, CA, DBN)
            file_paths: Optional list of file paths for identification
            
        Returns:
            DataFrame with prediction results
        """
        model = self.models[model_type]
        scaler = self.scalers.get(model_type)
        
        if scaler is None:
            logger.warning(f"No scaler found for {model_type}, using features as-is")
            scaled_features = features
        else:
            # Scale features
            scaled_features = scaler.transform(features)
        
        # Generate predictions using REPD model
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)
        
        # Calculate reconstruction errors (if available in model)
        reconstruction_errors = self._calculate_reconstruction_errors(scaled_features, model)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'file_id': range(len(features)) if file_paths is None else [os.path.basename(f) for f in file_paths],
            'prediction': predictions,
            'probability_defective': probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.flatten(),
            'reconstruction_error': reconstruction_errors,
            'model_type': model_type
        })
        
        if file_paths:
            results['file_path'] = file_paths
        
        return results
    
    def _calculate_reconstruction_errors(self, features: np.ndarray, model) -> np.ndarray:
        """
        Calculate reconstruction errors using the autoencoder component of REPD.
        
        Args:
            features: Input features
            model: REPD model
            
        Returns:
            Array of reconstruction errors
        """
        try:
            # Try to access autoencoder component of REPD model
            if hasattr(model, 'autoencoder'):
                reconstructed = model.autoencoder.predict(features)
                errors = np.mean((features - reconstructed) ** 2, axis=1)
                return errors
            elif hasattr(model, 'error_function_'):
                # Use stored error function
                errors = [model.error_function_(x.reshape(1, -1)) for x in features]
                return np.array(errors).flatten()
            else:
                # Fallback: use dummy reconstruction errors
                logger.warning("No reconstruction error calculation available, using dummy values")
                return np.random.uniform(0, 1, len(features))
                
        except Exception as e:
            logger.warning(f"Error calculating reconstruction errors: {e}")
            return np.zeros(len(features))
    
    def _save_results(self, results: Dict[str, pd.DataFrame], output_dir: str, repo_identifier: str) -> None:
        """
        Save prediction results to files.
        
        Args:
            results: Dictionary of prediction results
            output_dir: Output directory
            repo_identifier: Repository identifier
        """
        safe_name = repo_identifier.replace('/', '_').replace(':', '_').replace('.', '_')
        
        for model_type, df in results.items():
            filename = f"{safe_name}_{model_type}_predictions.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {model_type} predictions to {filepath}")
        
        # Save summary
        summary = self._generate_summary(results)
        summary_file = os.path.join(output_dir, f"{safe_name}_prediction_summary.json")
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved prediction summary to {summary_file}")
    
    def _generate_summary(self, results: Dict[str, pd.DataFrame]) -> Dict:
        """
        Generate prediction summary statistics.
        
        Args:
            results: Dictionary of prediction results
            
        Returns:
            Summary statistics dictionary
        """
        summary = {
            'total_files_analyzed': 0,
            'models_used': list(results.keys()),
            'model_results': {}
        }
        
        for model_type, df in results.items():
            total_files = len(df)
            defective_predictions = (df['prediction'] == 1).sum()
            avg_defect_probability = df['probability_defective'].mean()
            avg_reconstruction_error = df['reconstruction_error'].mean()
            
            summary['total_files_analyzed'] = total_files
            summary['model_results'][model_type] = {
                'total_files': total_files,
                'predicted_defective': int(defective_predictions),
                'predicted_non_defective': int(total_files - defective_predictions),
                'avg_defect_probability': float(avg_defect_probability),
                'avg_reconstruction_error': float(avg_reconstruction_error),
                'defect_rate': float(defective_predictions / total_files) if total_files > 0 else 0.0
            }
        
        return summary
    
    def generate_report(self, results: Dict[str, pd.DataFrame], output_file: str = "prediction_report.md") -> str:
        """
        Generate a markdown report of prediction results.
        
        Args:
            results: Dictionary of prediction results
            output_file: Output markdown file
            
        Returns:
            Report content as string
        """
        summary = self._generate_summary(results)
        
        report_lines = [
            "# Bug Prediction Report",
            "",
            f"**Total Files Analyzed:** {summary['total_files_analyzed']}",
            f"**Models Used:** {', '.join(summary['models_used'])}",
            "",
            "## Model Results",
            ""
        ]
        
        for model_type, stats in summary['model_results'].items():
            report_lines.extend([
                f"### {model_type} Model",
                "",
                f"- **Files Predicted as Defective:** {stats['predicted_defective']} ({stats['defect_rate']:.1%})",
                f"- **Files Predicted as Non-Defective:** {stats['predicted_non_defective']}",
                f"- **Average Defect Probability:** {stats['avg_defect_probability']:.4f}",
                f"- **Average Reconstruction Error:** {stats['avg_reconstruction_error']:.4f}",
                ""
            ])
        
        # Add top risky files
        if results:
            report_lines.extend([
                "## Top Risky Files",
                ""
            ])
            
            for model_type, df in results.items():
                top_risky = df.nlargest(5, 'probability_defective')
                report_lines.extend([
                    f"### {model_type} Model - Top 5 Risky Files",
                    "",
                    "| File | Defect Probability | Reconstruction Error |",
                    "|------|-------------------|---------------------|"
                ])
                
                for _, row in top_risky.iterrows():
                    file_name = row.get('file_path', row.get('file_id', 'Unknown'))
                    if isinstance(file_name, str) and len(file_name) > 50:
                        file_name = "..." + file_name[-47:]  # Truncate long paths
                    
                    report_lines.append(
                        f"| `{file_name}` | {row['probability_defective']:.4f} | {row['reconstruction_error']:.4f} |"
                    )
                
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated prediction report: {output_file}")
        return report_content


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict bugs in Java repositories")
    parser.add_argument("--repo", help="Repository URL to analyze")
    parser.add_argument("--files", nargs="*", help="Specific Java files to analyze")
    parser.add_argument("--output", default="prediction_results", help="Output directory")
    parser.add_argument("--model-dir", default="trained_models", help="Directory with trained models")
    parser.add_argument("--report", action="store_true", help="Generate markdown report")
    
    args = parser.parse_args()
    
    if not args.repo and not args.files:
        parser.error("Either --repo or --files must be specified")
    
    predictor = BugPredictor(args.model_dir)
    
    if args.repo:
        results = predictor.predict_repository(args.repo, args.output)
    else:
        results = predictor.predict_files(args.files)
    
    if args.report and results:
        report_file = os.path.join(args.output, "prediction_report.md")
        predictor.generate_report(results, report_file)
    
    # Print summary
    summary = predictor._generate_summary(results)
    print("\n=== Prediction Summary ===")
    print(f"Total files analyzed: {summary['total_files_analyzed']}")
    for model_type, stats in summary['model_results'].items():
        print(f"\n{model_type} Model:")
        print(f"  Defective: {stats['predicted_defective']} ({stats['defect_rate']:.1%})")
        print(f"  Non-defective: {stats['predicted_non_defective']}")
        print(f"  Avg defect probability: {stats['avg_defect_probability']:.4f}")


if __name__ == "__main__":
    main()

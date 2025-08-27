#!/usr/bin/env python3
"""
Simple Bug Predictor

A fallback bug prediction system that uses basic heuristics when the main REPD pipeline fails.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBugPredictor:
    """
    Simple bug predictor using basic heuristics.
    """
    
    def __init__(self):
        """Initialize the simple bug predictor."""
        self.feature_weights = {
            'lines_of_code': 0.3,
            'complexity': 0.4,
            'method_count': 0.2,
            'comment_ratio': -0.1  # Negative weight - more comments = less bugs
        }
    
    def predict_files(self, java_files: List[str], output_dir: str = "prediction_results") -> Dict[str, pd.DataFrame]:
        """
        Predict bugs in Java files using simple heuristics.
        
        Args:
            java_files: List of Java file paths
            output_dir: Output directory for results
            
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Predicting bugs in {len(java_files)} Java files using simple heuristics")
        
        results = []
        
        for file_path in java_files:
            try:
                # Extract basic features
                features = self._extract_basic_features(file_path)
                
                # Calculate bug probability using simple heuristics
                bug_probability = self._calculate_bug_probability(features)
                
                # Determine prediction (threshold at 0.5)
                prediction = 1 if bug_probability > 0.5 else 0
                
                results.append({
                    'file_path': file_path,
                    'file_id': os.path.basename(file_path),
                    'prediction': prediction,
                    'probability_defective': bug_probability,
                    'reconstruction_error': 0.0,  # Not applicable for simple predictor
                    'model_type': 'DA',
                    'lines_of_code': features['lines_of_code'],
                    'complexity': features['complexity'],
                    'method_count': features['method_count'],
                    'comment_ratio': features['comment_ratio']
                })
                
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                # Add default result for failed files
                results.append({
                    'file_path': file_path,
                    'file_id': os.path.basename(file_path),
                    'prediction': 0,
                    'probability_defective': 0.1,  # Low probability for failed files
                    'reconstruction_error': 0.0,
                    'model_type': 'DA',
                    'lines_of_code': 0,
                    'complexity': 0,
                    'method_count': 0,
                    'comment_ratio': 0
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "files_analysis_DA_predictions.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")
        
        # Save summary
        summary = self._generate_summary(df)
        summary_file = os.path.join(output_dir, "files_analysis_prediction_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
        
        return {'DA': df}
    
    def _extract_basic_features(self, file_path: str) -> Dict[str, float]:
        """
        Extract basic features from a Java file.
        
        Args:
            file_path: Path to Java file
            
        Returns:
            Dictionary of basic features
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Basic feature extraction
            lines_of_code = len(lines)
            method_count = content.count('public ') + content.count('private ') + content.count('protected ')
            comment_lines = sum(1 for line in lines if line.strip().startswith('//') or line.strip().startswith('/*'))
            comment_ratio = comment_lines / max(lines_of_code, 1)
            
            # Simple complexity measure (count of control structures)
            complexity = (
                content.count('if ') + 
                content.count('for ') + 
                content.count('while ') + 
                content.count('switch ') + 
                content.count('catch ') + 
                content.count('&&') + 
                content.count('||')
            )
            
            return {
                'lines_of_code': float(lines_of_code),
                'method_count': float(method_count),
                'comment_ratio': float(comment_ratio),
                'complexity': float(complexity)
            }
            
        except Exception as e:
            logger.warning(f"Error extracting features from {file_path}: {e}")
            return {
                'lines_of_code': 0.0,
                'method_count': 0.0,
                'comment_ratio': 0.0,
                'complexity': 0.0
            }
    
    def _calculate_bug_probability(self, features: Dict[str, float]) -> float:
        """
        Calculate bug probability using simple heuristics.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Bug probability between 0 and 1
        """
        # Normalize features to 0-1 range
        normalized_features = {
            'lines_of_code': min(features['lines_of_code'] / 1000.0, 1.0),  # Cap at 1000 lines
            'complexity': min(features['complexity'] / 50.0, 1.0),  # Cap at 50 complexity
            'method_count': min(features['method_count'] / 20.0, 1.0),  # Cap at 20 methods
            'comment_ratio': min(features['comment_ratio'], 1.0)  # Already 0-1
        }
        
        # Calculate weighted score
        score = sum(
            self.feature_weights[feature] * normalized_features[feature]
            for feature in self.feature_weights.keys()
        )
        
        # Convert to probability using sigmoid-like function
        probability = 1.0 / (1.0 + np.exp(-5 * (score - 0.3)))
        
        return min(max(probability, 0.0), 1.0)  # Ensure 0-1 range
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics.
        
        Args:
            df: DataFrame with predictions
            
        Returns:
            Summary dictionary
        """
        total_files = len(df)
        defective_predictions = (df['prediction'] == 1).sum()
        avg_defect_probability = df['probability_defective'].mean()
        
        return {
            'total_files_analyzed': total_files,
            'models_used': ['DA'],
            'model_results': {
                'DA': {
                    'total_files': total_files,
                    'predicted_defective': int(defective_predictions),
                    'predicted_non_defective': int(total_files - defective_predictions),
                    'avg_defect_probability': float(avg_defect_probability),
                    'avg_reconstruction_error': 0.0,
                    'defect_rate': float(defective_predictions / total_files) if total_files > 0 else 0.0
                }
            }
        }


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple bug prediction for Java files")
    parser.add_argument("--files", nargs="*", help="Java files to analyze")
    parser.add_argument("--output", default="prediction_results", help="Output directory")
    
    args = parser.parse_args()
    
    if not args.files:
        parser.error("--files must be specified")
    
    predictor = SimpleBugPredictor()
    results = predictor.predict_files(args.files, args.output)
    
    # Print summary
    summary = predictor._generate_summary(results['DA'])
    print("\n=== Simple Bug Prediction Summary ===")
    print(f"Total files analyzed: {summary['total_files_analyzed']}")
    print(f"Predicted defective: {summary['model_results']['DA']['predicted_defective']}")
    print(f"Predicted non-defective: {summary['model_results']['DA']['predicted_non_defective']}")
    print(f"Average defect probability: {summary['model_results']['DA']['avg_defect_probability']:.4f}")


if __name__ == "__main__":
    main() 
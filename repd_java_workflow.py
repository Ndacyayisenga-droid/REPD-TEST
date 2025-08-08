#!/usr/bin/env python3
"""
Simple REPD Integration Script

This script provides a simple interface for the REPD Java workflow,
similar to the original REPD-Workflow but adapted for Java semantic analysis.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup the REPD environment."""
    logger.info("Setting up REPD environment...")
    
    # Check required files
    required_files = [
        "trained_models/repd_model_DA.pkl",
        "trained_models/repd_model_CA.pkl", 
        "trained_models/repd_model_DBN.pkl",
        "semantic-dataset-creation/ASTEncoder-v1.2.jar"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error("Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False
    
    logger.info("Environment setup complete ✅")
    return True

def analyze_repository(repo_url, output_dir="analysis_results", models=None):
    """
    Analyze a Java repository for bug predictions.
    
    Args:
        repo_url: Repository URL to analyze
        output_dir: Output directory for results
        models: List of models to use (default: all)
    """
    logger.info(f"Analyzing repository: {repo_url}")
    
    if models is None:
        models = ["DA", "CA", "DBN"]
    
    try:
        # Import and use the bug prediction pipeline
        from bug_prediction_pipeline import BugPredictor
        
        predictor = BugPredictor()
        results = predictor.predict_repository(repo_url, output_dir)
        
        if results:
            # Generate report
            report = predictor.generate_report(results, os.path.join(output_dir, "analysis_report.md"))
            
            logger.info("Analysis completed successfully ✅")
            logger.info(f"Results saved to: {output_dir}")
            
            # Print summary
            summary = predictor._generate_summary(results)
            print(f"\n📊 Analysis Summary:")
            print(f"Files analyzed: {summary['total_files_analyzed']}")
            
            for model_type, stats in summary['model_results'].items():
                print(f"\n{model_type} Model:")
                print(f"  Defective files: {stats['predicted_defective']} ({stats['defect_rate']:.1%})")
                print(f"  Average defect probability: {stats['avg_defect_probability']:.4f}")
            
            return True
        else:
            logger.error("Analysis failed - no results generated")
            return False
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False

def analyze_files(file_paths, output_dir="file_analysis_results"):
    """
    Analyze specific Java files.
    
    Args:
        file_paths: List of Java file paths
        output_dir: Output directory for results
    """
    logger.info(f"Analyzing {len(file_paths)} Java files")
    
    try:
        from bug_prediction_pipeline import BugPredictor
        
        predictor = BugPredictor()
        results = predictor.predict_files(file_paths)
        
        if results:
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate simple report
            report_lines = ["# Java File Analysis Report", ""]
            
            for model_type, df in results.items():
                report_lines.extend([
                    f"## {model_type} Model Results",
                    "",
                    "| File | Defect Probability | Prediction |",
                    "|------|-------------------|------------|"
                ])
                
                for _, row in df.iterrows():
                    file_name = os.path.basename(row.get('file_path', row.get('file_id', 'Unknown')))
                    prob = row['probability_defective']
                    pred = "🔴 Defective" if row['prediction'] == 1 else "✅ Clean"
                    
                    report_lines.append(f"| `{file_name}` | {prob:.4f} | {pred} |")
                
                report_lines.append("")
            
            # Save report
            report_file = os.path.join(output_dir, "file_analysis_report.md")
            with open(report_file, 'w') as f:
                f.write("\n".join(report_lines))
            
            logger.info(f"File analysis completed ✅")
            logger.info(f"Report saved to: {report_file}")
            
            return True
        else:
            logger.error("File analysis failed - no results generated")
            return False
            
    except Exception as e:
        logger.error(f"File analysis failed: {e}")
        return False

def extract_features_only(repo_url, output_dir="extracted_features"):
    """
    Extract features from repository without prediction.
    
    Args:
        repo_url: Repository URL
        output_dir: Output directory
    """
    logger.info(f"Extracting features from: {repo_url}")
    
    try:
        from java_feature_extractor import JavaFeatureExtractor
        
        extractor = JavaFeatureExtractor()
        features = extractor.extract_features_from_repository(repo_url, output_dir)
        
        if features:
            logger.info("Feature extraction completed ✅")
            print(f"\n📊 Extracted Features:")
            for feature_type, feature_data in features.items():
                print(f"  {feature_type}: {feature_data.shape}")
            
            return True
        else:
            logger.error("Feature extraction failed")
            return False
            
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return False

def validate_models():
    """Validate all trained models."""
    logger.info("Validating REPD models...")
    
    try:
        from model_manager import ModelManager
        
        manager = ModelManager()
        results = manager.validate_models()
        
        print("\n🔧 Model Validation Results:")
        for model_type, result in results.items():
            status = result.get('status', 'unknown')
            emoji = "✅" if status == "validated" else "⚠️"
            print(f"  {model_type}: {emoji} {status}")
            
            if 'prediction_distribution' in result:
                dist = result['prediction_distribution']
                total = dist['defective'] + dist['non_defective']
                print(f"    Test predictions: {dist['defective']} defective, {dist['non_defective']} clean (total: {total})")
        
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="REPD Java Workflow - Bug prediction for Java repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze entire repository
  python repd_java_workflow.py --repo https://github.com/example/java-project.git
  
  # Analyze specific files
  python repd_java_workflow.py --files src/main/java/Example.java src/test/java/Test.java
  
  # Extract features only
  python repd_java_workflow.py --extract-features --repo https://github.com/example/project.git
  
  # Validate models
  python repd_java_workflow.py --validate-models
        """
    )
    
    # Main actions
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo", help="Repository URL to analyze")
    group.add_argument("--files", nargs="+", help="Specific Java files to analyze")
    group.add_argument("--extract-features", action="store_true", help="Extract features only (requires --repo)")
    group.add_argument("--validate-models", action="store_true", help="Validate trained models")
    
    # Options
    parser.add_argument("--output", default="repd_results", help="Output directory (default: repd_results)")
    parser.add_argument("--models", nargs="+", choices=["DA", "CA", "DBN"], help="Models to use (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    success = False
    
    try:
        if args.validate_models:
            success = validate_models()
            
        elif args.extract_features:
            if not args.repo:
                parser.error("--extract-features requires --repo")
            success = extract_features_only(args.repo, args.output)
            
        elif args.repo:
            success = analyze_repository(args.repo, args.output, args.models)
            
        elif args.files:
            # Verify files exist
            existing_files = [f for f in args.files if os.path.exists(f)]
            if not existing_files:
                logger.error("No valid Java files found")
                sys.exit(1)
            
            if len(existing_files) != len(args.files):
                logger.warning(f"Some files not found. Analyzing {len(existing_files)}/{len(args.files)} files.")
            
            success = analyze_files(existing_files, args.output)
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    
    if success:
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        sys.exit(0)
    else:
        print(f"\n❌ Analysis failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Java Feature Extraction Pipeline

This module implements feature extraction from Java files using semantic analysis,
adapted from REPD-Workflow logic to work with Java code and semantic datasets.
"""

import os
import subprocess
import tempfile
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JavaFeatureExtractor:
    """
    Feature extractor for Java files using AST-based semantic analysis.
    Adapted from REPD-Workflow logic for Java code.
    """
    
    def __init__(self, ast_encoder_path: str = "semantic-dataset-creation/ASTEncoder-v1.2.jar"):
        """
        Initialize the Java feature extractor.
        
        Args:
            ast_encoder_path: Path to the AST encoder JAR file
        """
        self.ast_encoder_path = ast_encoder_path
        self.config_path = "semantic-dataset-creation/config"
        self.temp_dir = None
        
    def extract_features_from_repository(self, repo_url: str, output_dir: str = "extracted_features") -> Dict[str, np.ndarray]:
        """
        Extract features from a Java repository.
        
        Args:
            repo_url: URL of the Git repository
            output_dir: Directory to save extracted features
            
        Returns:
            Dictionary containing extracted features for each feature type
        """
        logger.info(f"Extracting features from repository: {repo_url}")
        
        # Create temporary directory for repository
        self.temp_dir = tempfile.mkdtemp()
        repo_dir = os.path.join(self.temp_dir, "repo")
        
        try:
            # Clone repository
            self._clone_repository(repo_url, repo_dir)
            
            # Find Java files
            java_files = self._find_java_files(repo_dir)
            logger.info(f"Found {len(java_files)} Java files")
            
            # Extract AST vectors
            ast_vectors = self._extract_ast_vectors(java_files)
            
            # Generate semantic features using all extractors
            features = self._generate_semantic_features(ast_vectors)
            
            # Save features
            os.makedirs(output_dir, exist_ok=True)
            self._save_features(features, output_dir, repo_url)
            
            return features
            
        finally:
            # Clean up temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                
    def extract_features_from_files(self, java_files: List[str], use_existing: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract features from a list of Java files.
        
        Args:
            java_files: List of paths to Java files
            use_existing: If True, attempt to load preprocessed features; if False, force AST extraction
            
        Returns:
            Dictionary containing extracted features for each feature type
        """
        logger.info(f"Extracting features from {len(java_files)} Java files")
        
        # Try to use existing features first (only if explicitly allowed)
        if use_existing:
            existing_features = self._load_existing_features()
            if existing_features:
                logger.info("Using existing pre-processed features")
                return existing_features
        
        # Fallback to AST extraction if no existing features or not allowed
        logger.info("Generating features via AST extraction for provided files")
        ast_vectors = self._extract_ast_vectors(java_files)
        
        # Generate semantic features
        features = self._generate_semantic_features(ast_vectors)
        
        return features
    
    def _clone_repository(self, repo_url: str, target_dir: str) -> None:
        """Clone a Git repository."""
        try:
            subprocess.run([
                "git", "clone", "--depth", "1", repo_url, target_dir
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
            
    def _find_java_files(self, directory: str) -> List[str]:
        """Find all Java files in a directory."""
        java_files = []
        for root, dirs, files in os.walk(directory):
            # Skip test directories and build artifacts
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['target', 'build', 'bin']]
            
            for file in files:
                if file.endswith('.java'):
                    java_files.append(os.path.join(root, file))
                    
        return java_files
    
    def _extract_ast_vectors(self, java_files: List[str]) -> List[List[int]]:
        """
        Extract AST vectors from Java files using the AST encoder.
        
        Args:
            java_files: List of Java file paths
            
        Returns:
            List of AST vectors (tokenized representations)
        """
        logger.info("Extracting AST vectors from Java files")
        
        vectors = []
        
        for java_file in java_files:
            try:
                # Use the AST encoder to extract tokens
                vector = self._extract_single_ast_vector(java_file)
                if vector:
                    vectors.append(vector)
            except Exception as e:
                logger.warning(f"Failed to extract AST vector from {java_file}: {e}")
                
        logger.info(f"Extracted {len(vectors)} AST vectors")
        return vectors
    
    def _extract_single_ast_vector(self, java_file: str) -> Optional[List[int]]:
        """
        Extract AST vector from a single Java file.
        
        Args:
            java_file: Path to Java file
            
        Returns:
            AST vector as list of integers
        """
        try:
            # Create temporary output file
            temp_output = tempfile.mktemp(suffix='.txt')
            
            # Run AST encoder
            cmd = [
                "java", "-jar", self.ast_encoder_path,
                java_file,
                temp_output,
                os.path.join(self.config_path, "parser.properties")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(temp_output):
                # Read the extracted tokens
                with open(temp_output, 'r') as f:
                    tokens = f.read().strip().split()
                    vector = [int(token) for token in tokens if token.isdigit()]
                    
                os.unlink(temp_output)
                return vector
            else:
                logger.warning(f"AST encoder failed for {java_file}: {result.stderr}")
                return None
                
        except Exception as e:
            logger.warning(f"Error extracting AST vector from {java_file}: {e}")
            return None
    
    def _generate_semantic_features(self, ast_vectors: List[List[int]]) -> Dict[str, np.ndarray]:
        """
        Generate semantic features using different extraction methods.
        
        Args:
            ast_vectors: List of AST vectors
            
        Returns:
            Dictionary with features for each extractor type
        """
        logger.info("Generating semantic features")
        
        if not ast_vectors:
            logger.warning("No AST vectors available for feature extraction")
            return {}
        
        # Import extractors
        import sys
        sys.path.append('semantic-dataset-creation')
        from extractor import DeepAutoencoder
        
        # Prepare data (pad to same length)
        prepared_data = self._prepare_data(ast_vectors)
        
        features = {}
        
        try:
            # Deep Autoencoder features only
            logger.info("Extracting DA features")
            da_extractor = DeepAutoencoder()
            da_features = da_extractor.get_features(prepared_data, None)
            features['DA'] = np.array([x.flatten() for x in da_features])
            
        except Exception as e:
            logger.error(f"DA feature extraction failed: {e}")
        
        return features
    
    def _load_existing_features(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Load existing pre-processed features if available.
        
        Returns:
            Dictionary of features if available, None otherwise
        """
        try:
            features = {}
            
            # Check for DA features only
            da_path = "openj9_metrics_DA_features.npy"
            if os.path.exists(da_path):
                da_features = np.load(da_path)
                features['DA'] = da_features
                logger.info(f"Loaded DA features: {da_features.shape}")
                return features
            else:
                logger.info("No existing DA features found")
                return None
                
        except Exception as e:
            logger.warning(f"Error loading existing features: {e}")
            return None
    
    def _prepare_data(self, data: List[List[int]]) -> np.ndarray:
        """
        Prepare AST vectors by padding to same length.
        
        Args:
            data: List of AST vectors
            
        Returns:
            Padded numpy array
        """
        if not data:
            return np.array([])
            
        max_len = max([len(x) for x in data])
        if max_len % 8:
            max_len += (8 - (max_len % 8))
            
        padded_data = np.array([
            np.pad(np.array(x), (0, max_len - len(x)), 'constant') 
            for x in data
        ])
        
        return padded_data
    
    def _save_features(self, features: Dict[str, np.ndarray], output_dir: str, repo_identifier: str) -> None:
        """
        Save extracted features to files.
        
        Args:
            features: Dictionary of features
            output_dir: Output directory
            repo_identifier: Repository identifier for naming
        """
        # Create a safe filename from repository URL
        safe_name = repo_identifier.replace('/', '_').replace(':', '_').replace('.', '_')
        
        for feature_type, feature_data in features.items():
            filename = f"{safe_name}_{feature_type}_features.npy"
            filepath = os.path.join(output_dir, filename)
            np.save(filepath, feature_data)
            logger.info(f"Saved {feature_type} features to {filepath}")
            
        # Save metadata
        metadata = {
            'repository': repo_identifier,
            'feature_types': list(features.keys()),
            'feature_shapes': {k: list(v.shape) for k, v in features.items()},
            'total_files': len(features.get('DA', [])) if 'DA' in features else 0
        }
        
        metadata_file = os.path.join(output_dir, f"{safe_name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract features from Java repositories")
    parser.add_argument("--repo", required=True, help="Repository URL or path")
    parser.add_argument("--output", default="extracted_features", help="Output directory")
    parser.add_argument("--files", nargs="*", help="Specific Java files to process")
    
    args = parser.parse_args()
    
    extractor = JavaFeatureExtractor()
    
    if args.files:
        # Process specific files
        features = extractor.extract_features_from_files(args.files)
    else:
        # Process repository
        features = extractor.extract_features_from_repository(args.repo, args.output)
    
    print(f"Feature extraction completed. Features extracted:")
    for feature_type, feature_data in features.items():
        print(f"  {feature_type}: {feature_data.shape}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple test script to verify bug prediction pipeline components.
"""

import os
import sys
import subprocess

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        import numpy
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
        
    try:
        import pandas
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
        
    try:
        import sklearn
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
        
    try:
        import tensorflow
        print("✓ tensorflow imported successfully")
    except ImportError as e:
        print(f"✗ tensorflow import failed: {e}")
        return False
        
    try:
        import torch
        print("✓ torch imported successfully")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False
        
    return True

def test_files_exist():
    """Test if required files exist."""
    print("\nTesting file existence...")
    
    required_files = [
        "semantic-dataset-creation/ASTEncoder-v1.2.jar",
        "semantic-dataset-creation/config/parser.properties",
        "trained_models/repd_model_DA.pkl",
        "trained_models/training_results.pkl"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
            
    return all_exist

def test_java():
    """Test if Java is available."""
    print("\nTesting Java...")
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Java is available")
            print(f"  Version: {result.stderr.split()[2]}")
            return True
        else:
            print("✗ Java command failed")
            return False
    except FileNotFoundError:
        print("✗ Java not found in PATH")
        return False

def test_ast_encoder():
    """Test if AST encoder can be executed."""
    print("\nTesting AST encoder...")
    try:
        # Create a simple test Java file
        test_java = """
        public class Test {
            public static void main(String[] args) {
                System.out.println("Hello World");
            }
        }
        """
        
        with open("Test.java", "w") as f:
            f.write(test_java)
            
        # Try to run AST encoder
        cmd = [
            "java", "-jar", "semantic-dataset-creation/ASTEncoder-v1.2.jar",
            "Test.java",
            "test_output.txt",
            "semantic-dataset-creation/config/parser.properties"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ AST encoder executed successfully")
            if os.path.exists("test_output.txt"):
                print("✓ Output file created")
                with open("test_output.txt", "r") as f:
                    content = f.read()
                    print(f"  Output length: {len(content)} characters")
            else:
                print("✗ Output file not created")
        else:
            print(f"✗ AST encoder failed: {result.stderr}")
            
        # Clean up
        if os.path.exists("Test.java"):
            os.remove("Test.java")
        if os.path.exists("test_output.txt"):
            os.remove("test_output.txt")
            
    except Exception as e:
        print(f"✗ AST encoder test failed: {e}")

def test_bug_prediction_pipeline():
    """Test if bug prediction pipeline can be imported and initialized."""
    print("\nTesting bug prediction pipeline...")
    try:
        from bug_prediction_pipeline import BugPredictor
        print("✓ BugPredictor imported successfully")
        
        predictor = BugPredictor()
        print("✓ BugPredictor initialized successfully")
        
        print(f"  Available models: {list(predictor.models.keys())}")
        print(f"  Available scalers: {list(predictor.scalers.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Bug prediction pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running workflow component tests...\n")
    
    tests = [
        test_imports,
        test_files_exist,
        test_java,
        test_bug_prediction_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    test_names = ["Imports", "Files", "Java", "Pipeline"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASS" if result else "FAIL"
        print(f"{name:12}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    
    if not all_passed:
        print("\nRecommendations:")
        if not results[0]:  # Imports failed
            print("- Install missing Python dependencies")
        if not results[1]:  # Files missing
            print("- Ensure all required files are present in the repository")
        if not results[2]:  # Java not available
            print("- Install Java and ensure it's in PATH")
        if not results[3]:  # Pipeline failed
            print("- Check model files and dependencies")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 
#!/usr/bin/env python
"""
Test runner for the football prediction system.
Executes both black box and white box tests.
"""
import unittest
import os
import sys
import glob

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def run_tests(test_type=None):
    """
    Run all tests or a specific type of tests.
    
    Args:
        test_type (str, optional): Type of tests to run ('black_box', 'white_box', or None for all)
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Initialize a test loader
    loader = unittest.TestLoader()
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Function to add tests from a directory
    def add_tests_from_directory(directory):
        print(f"Loading tests from {directory}...")
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist.")
            return
        
        # Find all test Python files
        test_files = glob.glob(os.path.join(directory, 'test_*.py'))
        for test_file in test_files:
            module_name = os.path.basename(test_file)[:-3]  # Remove .py extension
            
            # Construct the import path
            import_path = f"tests.{os.path.basename(directory)}.{module_name}"
            try:
                # Import the module and add its tests
                module = __import__(import_path, fromlist=['*'])
                tests = loader.loadTestsFromModule(module)
                test_suite.addTest(tests)
                print(f"Added tests from {import_path}")
            except Exception as e:
                print(f"Error loading tests from {import_path}: {str(e)}")
    
    # Add tests based on the test type
    if test_type == 'black_box':
        add_tests_from_directory(os.path.join(os.path.dirname(__file__), 'black_box'))
    elif test_type == 'white_box':
        add_tests_from_directory(os.path.join(os.path.dirname(__file__), 'white_box'))
    else:
        # Run both types of tests
        add_tests_from_directory(os.path.join(os.path.dirname(__file__), 'black_box'))
        add_tests_from_directory(os.path.join(os.path.dirname(__file__), 'white_box'))
    
    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    print(f"\n{'='*70}")
    if test_type:
        print(f"RUNNING {test_type.upper()} TESTS")
    else:
        print("RUNNING ALL TESTS")
    print(f"{'='*70}\n")
    
    result = test_runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Return True if all tests passed
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run tests for the football prediction system')
    parser.add_argument('--type', choices=['black_box', 'white_box', 'all'], 
                        default='all', help='Type of tests to run')
    
    args = parser.parse_args()
    
    # Run tests based on command line arguments
    test_type = None if args.type == 'all' else args.type
    success = run_tests(test_type)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 
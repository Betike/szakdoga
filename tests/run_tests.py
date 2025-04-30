import unittest
import os
import sys
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def run_tests(test_type=None):
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    def add_tests_from_directory(directory):
        print(f"Loading tests from {directory}...")
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist.")
            return
        
        test_files = glob.glob(os.path.join(directory, 'test_*.py'))
        for test_file in test_files:
            module_name = os.path.basename(test_file)[:-3]
            
            import_path = f"tests.{os.path.basename(directory)}.{module_name}"
            try:
                module = __import__(import_path, fromlist=['*'])
                tests = loader.loadTestsFromModule(module)
                test_suite.addTest(tests)
                print(f"Added tests from {import_path}")
            except Exception as e:
                print(f"Error loading tests from {import_path}: {str(e)}")
    
    if test_type == 'black_box':
        add_tests_from_directory(os.path.join(os.path.dirname(__file__), 'black_box'))
    elif test_type == 'white_box':
        add_tests_from_directory(os.path.join(os.path.dirname(__file__), 'white_box'))
    else:
        add_tests_from_directory(os.path.join(os.path.dirname(__file__), 'black_box'))
        add_tests_from_directory(os.path.join(os.path.dirname(__file__), 'white_box'))
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    print(f"\n{'='*70}")
    if test_type:
        print(f"RUNNING {test_type.upper()} TESTS")
    else:
        print("RUNNING ALL TESTS")
    print(f"{'='*70}\n")
    
    result = test_runner.run(test_suite)
    
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for the football prediction system')
    parser.add_argument('--type', choices=['black_box', 'white_box', 'all'], 
                        default='all', help='Type of tests to run')
    
    args = parser.parse_args()
    
    test_type = None if args.type == 'all' else args.type
    success = run_tests(test_type)
    
    sys.exit(0 if success else 1) 
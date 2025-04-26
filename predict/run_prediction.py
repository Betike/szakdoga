#!/usr/bin/env python
"""
Wrapper script for running predictions with the correct environment
"""
import os
import sys
import subprocess
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run prediction with correct environment')
    parser.add_argument('--model', choices=['xgboost', 'random_forest', 'pytorch', 'ensemble'], 
                       required=True, help='Model to use for prediction')
    parser.add_argument('--home', type=str, required=True, help='Home team name')
    parser.add_argument('--away', type=str, required=True, help='Away team name')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the root directory (parent of the script directory)
    root_dir = os.path.dirname(script_dir)
    
    # Set up the correct environment
    os.environ['PYTHONPATH'] = root_dir
    
    # Determine the script to run
    predictor_script = os.path.join(script_dir, f"predict_with_{args.model}.py")
    if args.model == 'ensemble':
        predictor_script = os.path.join(script_dir, "predict_ensemble.py")
    
    # Check if the script exists
    if not os.path.exists(predictor_script):
        error = f"Error: Predictor script not found: {predictor_script}"
        if args.json:
            print(json.dumps({"error": error}))
        else:
            print(error)
        return 1
    
    # Build the command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        predictor_script,
        "--single-match",
        "--home", args.home,
        "--away", args.away
    ]
    
    if args.json:
        cmd.append("--json")
        # In JSON mode, suppress all debug output
        os.environ["PYTHONUNBUFFERED"] = "0"
    else:
        # Print debug info
        print(f"Running prediction with {args.model} model")
        print(f"Script: {predictor_script}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        # Use the project root as the working directory
        result = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            if args.json:
                # Try to extract JSON error if possible
                try:
                    # Look for a JSON object in the output
                    output = result.stdout.strip()
                    json_start = output.find('{')
                    if json_start >= 0:
                        error_data = json.loads(output[json_start:])
                        print(json.dumps(error_data))
                    else:
                        print(json.dumps({"error": result.stderr or f"Command failed with exit code {result.returncode}"}))
                    return result.returncode
                except Exception as e:
                    print(json.dumps({"error": f"Failed to parse output: {str(e)}, stdout: {result.stdout}, stderr: {result.stderr}"}))
                    return result.returncode
            else:
                print(f"Error: Command failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"Error details: {result.stderr}")
                return result.returncode
        
        # Process output to extract only the JSON part if in JSON mode
        if args.json:
            try:
                output = result.stdout.strip()
                # Find the first '{' character, which should be the start of the JSON object
                json_start = output.find('{')
                if json_start >= 0:
                    # Extract only the JSON part
                    json_str = output[json_start:]
                    # Parse to ensure it's valid JSON
                    json_obj = json.loads(json_str)
                    # Print only the JSON output
                    print(json.dumps(json_obj))
                else:
                    # No JSON found, return an error
                    print(json.dumps({"error": "No JSON output found in script result"}))
                    return 1
            except Exception as e:
                print(json.dumps({"error": f"Failed to parse JSON output: {str(e)}"}))
                return 1
        else:
            # In non-JSON mode, output the full result
            print(result.stdout)
        
        return 0
    
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error executing prediction: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
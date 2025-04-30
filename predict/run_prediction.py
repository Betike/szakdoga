import os
import sys
import subprocess
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['xgboost', 'random_forest', 'pytorch', 'ensemble'], 
                       required=True, help='Model to use')
    parser.add_argument('--home', type=str, required=True, help='Home team name')
    parser.add_argument('--away', type=str, required=True, help='Away team name')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    root_dir = os.path.dirname(script_dir)
    
    os.environ['PYTHONPATH'] = root_dir
    
    predictor_script = os.path.join(script_dir, f"predict_with_{args.model}.py")
    if args.model == 'ensemble':
        predictor_script = os.path.join(script_dir, "predict_ensemble.py")
    
    if not os.path.exists(predictor_script):
        error = f"Error: Predictor script not found: {predictor_script}"
        if args.json:
            print(json.dumps({"error": error}))
        else:
            print(error)
        return 1
    
    cmd = [
        sys.executable,
        predictor_script,
        "--single-match",
        "--home", args.home,
        "--away", args.away
    ]
    
    if args.json:
        cmd.append("--json")
        os.environ["PYTHONUNBUFFERED"] = "0"
    else:
        print(f"Running prediction with {args.model} model")
    
    try:
        result = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True, check=False)
        
        if args.debug:
            with open('debug_output.txt', 'w') as f:
                f.write(f"Raw output: {result.stdout}\n")
                f.write(f"Raw stderr: {result.stderr}\n")
        
        if result.returncode != 0:
            if args.json:
                try:
                    output = result.stdout.strip()
                    json_start = output.find('{')
                    if json_start >= 0:
                        error_data = json.loads(output[json_start:])
                        print(json.dumps(error_data))
                    else:
                        print(json.dumps({"error": result.stderr or f"Failed with exit code {result.returncode}"}))
                    return result.returncode
                except Exception as e:
                    print(json.dumps({"error": f"Failed to parse output: {str(e)}, stdout: {result.stdout}, stderr: {result.stderr}"}))
                    return result.returncode
            else:
                print(f"Error: Failed with exit code {result.returncode}")
                if result.stderr:
                    print(f"Error details: {result.stderr}")
                return result.returncode
        
        if args.json:
            try:
                output = result.stdout.strip()
                json_start = output.find('{')
                json_end = output.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = output[json_start:json_end]
                    json_obj = json.loads(json_str)
                    print(json.dumps(json_obj))
                else:
                    print(json.dumps({"error": "No valid JSON output found in script result"}))
                    return 1
            except Exception as e:
                print(json.dumps({"error": f"Failed to parse JSON output: {str(e)}"}))
                return 1
        else:
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
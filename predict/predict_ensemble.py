import argparse
import subprocess
import json
import os
import traceback
import pandas as pd
from datetime import datetime
import sys

MODELS = ["xgboost", "random_forest", "pytorch"]

def run_individual_predictor(predictor_type, home_team, away_team):
    wrapper_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_prediction.py")
    
    try:
        print(f"Running {predictor_type} predictor...")
        print(f"Using wrapper script: {wrapper_script}")
        print(f"Script exists: {os.path.exists(wrapper_script)}")
        
        cmd = [
            sys.executable,
            wrapper_script,
            "--model", predictor_type,
            "--home", home_team,
            "--away", away_team,
            "--json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print(f"Warning: {predictor_type} predictor exited with code {result.returncode}")
            print(f"Stderr: {result.stderr}")
            print(f"Stdout: {result.stdout}")
            
            try:
                error_data = json.loads(result.stdout)
                if "error" in error_data:
                    return {"error": error_data["error"]}
            except:
                pass
                
            return {"error": result.stderr.strip() or f"{predictor_type} predictor failed"}
        
        try:
            result_data = json.loads(result.stdout)
            return result_data
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON output from {predictor_type} predictor")
            print(f"Output: {result.stdout}")
            return {"error": f"Invalid JSON output from {predictor_type} predictor"}
    
    except Exception as e:
        print(f"Error running {predictor_type} predictor: {str(e)}")
        return {"error": str(e)}

def ensemble_predict(home_team, away_team):
    
    results = {}
    errors = {}
    successful_models = 0
    
    print(f"\nPredicting result for {home_team} vs {away_team} using ensemble model")
    
    for model in MODELS:
        result = run_individual_predictor(model, home_team, away_team)
        if "error" in result:
            errors[model] = result["error"]
            print(f"  - {model.upper()} predictor failed: {result['error']}")
        else:
            results[model] = result
            successful_models += 1
            print(f"  - {model.upper()} predictor successful: predicted {result['prediction']}")
    
    if not results:
        error_details = "\n".join([f"{model}: {error}" for model, error in errors.items()])
        return {
            "error": f"All predictors failed. Details:\n{error_details}"
        }
    
    all_probabilities = {}
    for outcome in ["Home win", "Draw", "Away win"]:
        probs = [model_result["probabilities"].get(outcome, 0) 
                for model_result in results.values()]
        all_probabilities[outcome] = sum(probs) / len(probs)
    
    predicted_outcome = max(all_probabilities, key=all_probabilities.get)
    
    outcome_map = {"Home win": "H", "Draw": "D", "Away win": "A"}
    final_prediction = outcome_map.get(predicted_outcome, "Unknown")
    
    failed_models_info = {}
    if errors:
        failed_models_info = {
            "failed_models": errors,
            "warning": f"{len(errors)} out of {len(MODELS)} models failed. Prediction is based on {successful_models} models."
        }
    
    return {
        "prediction": final_prediction,
        "probabilities": all_probabilities,
        "individual_models": results,
        "models_used": successful_models,
        "total_models": len(MODELS),
        **failed_models_info
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Predict football match outcomes using an ensemble of models')
    parser.add_argument('--single-match', action='store_true', help='Predict a single match')
    parser.add_argument('--home', type=str, help='Home team name')
    parser.add_argument('--away', type=str, help='Away team name')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--file', type=str, help='File with matches to predict')
    return parser.parse_args()

def process_batch_file(file_path, json_output=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    
    matches = pd.read_csv(file_path)
    required_columns = ['HomeTeam', 'AwayTeam']
    
    if not all(col in matches.columns for col in required_columns):
        raise ValueError(f"Input file must contain columns: {required_columns}")
    
    identical_teams = matches[matches['HomeTeam'] == matches['AwayTeam']]
    if len(identical_teams) > 0 and not json_output:
        print(f"Found {len(identical_teams)} matches with identical home and away teams.")
        print("These matches will be skipped:")
        for idx, row in identical_teams.iterrows():
            print(f"- {row['HomeTeam']} vs {row['AwayTeam']}")

        matches = matches[matches['HomeTeam'] != matches['AwayTeam']]
        print(f"Proceeding with {len(matches)} valid matches.")
    
    matches['PredictedResult'] = None
    matches['HomeWinProb'] = None
    matches['DrawProb'] = None
    matches['AwayWinProb'] = None
    
    if not json_output:
        print(f"\nPredicting outcomes for {len(matches)} matches...")
    
    for idx, row in matches.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        try:
            if home_team == away_team:
                continue
                
            result = ensemble_predict(home_team, away_team)
            
            if "error" not in result:
                matches.at[idx, 'PredictedResult'] = result["prediction"]
                matches.at[idx, 'HomeWinProb'] = result["probabilities"].get("Home win", 0)
                matches.at[idx, 'DrawProb'] = result["probabilities"].get("Draw", 0)
                matches.at[idx, 'AwayWinProb'] = result["probabilities"].get("Away win", 0)
        except Exception as e:
            if not json_output:
                print(f"Error predicting match {home_team} vs {away_team}: {e}")
    
    return matches

if __name__ == "__main__":
    try:
        args = parse_args()
        
        if args.single_match and args.home and args.away:
            home_team = args.home
            away_team = args.away
            
            if home_team == away_team:
                error_msg = "Home team and away team cannot be identical"
                if args.json:
                    print(json.dumps({"error": error_msg}))
                else:
                    print(f"Error: {error_msg}")
                exit(1)
            
            try:
                result = ensemble_predict(home_team, away_team)
                
                if "error" not in result:
                    if args.json:
                        print(json.dumps({
                            "prediction": result["prediction"],
                            "probabilities": result["probabilities"]
                        }))
                    else:
                        print(f"\nEnsemble prediction for {home_team} vs {away_team}: {result['prediction']}")
                        
                        print("\nCombined probabilities:")
                        for outcome, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
                            print(f"{outcome}: {prob:.2%}")
                        
                        print("\nIndividual model predictions:")
                        for model, model_result in result["individual_models"].items():
                            print(f"  {model.upper()}: {model_result['prediction']} " +
                                  f"(Home: {model_result['probabilities'].get('Home win', 0):.2%}, " +
                                  f"Draw: {model_result['probabilities'].get('Draw', 0):.2%}, " +
                                  f"Away: {model_result['probabilities'].get('Away win', 0):.2%})")
                else:
                    if args.json:
                        print(json.dumps({"error": result["error"]}))
                    else:
                        print(f"Error: {result['error']}")
            except Exception as e:
                if args.json:
                    print(json.dumps({"error": str(e)}))
                else:
                    print(f"Error: {e}")
        
        elif args.file:
            try:
                matches = process_batch_file(args.file, args.json)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"ensemble_predictions_{timestamp}.csv"
                matches.to_csv(output_file, index=False)
                
                if not args.json:
                    print(f"\nPredictions completed and saved to {output_file}")
                    
                else:
                    print(json.dumps(matches.to_dict(orient='records')))
            except Exception as e:
                if args.json:
                    print(json.dumps({"error": str(e)}))
                else:
                    print(f"Error processing file: {e}")
        
        else:
            if not args.json:
                print("Ensemble Model")
                
                choice = input("Single match (1) / Multiple matches from a file(2): ")
                
                if choice == '1':
                    print("\nEnter team names for prediction:")
                    home_team = input("Home team: ")
                    away_team = input("Away team: ")
                    
                    if home_team == away_team:
                        print("\nError: Home team and away team cannot be identical")
                    else:
                        print("\nRunning ensemble prediction...")
                        try:
                            result = ensemble_predict(home_team, away_team)
                            
                            if "error" not in result:
                                print(f"\nEnsemble prediction for {home_team} vs {away_team}: {result['prediction']}")
                                
                                print("\nCombined probabilities:")
                                for outcome, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
                                    print(f"{outcome}: {prob:.2%}")
                                
                                print("\nIndividual model predictions:")
                                for model, model_result in result["individual_models"].items():
                                    print(f"  {model.upper()}: {model_result['prediction']} " +
                                          f"(Home: {model_result['probabilities'].get('Home win', 0):.2%}, " +
                                          f"Draw: {model_result['probabilities'].get('Draw', 0):.2%}, " +
                                          f"Away: {model_result['probabilities'].get('Away win', 0):.2%})")
                            else:
                                print(f"\nError: {result['error']}")
                        except Exception as e:
                            print(f"\nError: {e}")
                
                elif choice == '2':
                    file_path = input("Enter path to CSV file containing matches (format: HomeTeam,AwayTeam): ")
                    
                    try:
                        matches = process_batch_file(file_path)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = f"ensemble_predictions_{timestamp}.csv"
                        matches.to_csv(output_file, index=False)
                        
                        print(f"\nPredictions completed and saved to {output_file}")
                        
                    except Exception as e:
                        print(f"Error: {e}")
                
                else:
                    print("Invalid choice. Please run again and select 1 or 2.")
            else:
                print(json.dumps({"error": "Missing required arguments. Use --single-match with --home and --away, or use --file"}))
    
    except Exception as e:
        if args.json if 'args' in locals() else False:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}")
            traceback.print_exc() 
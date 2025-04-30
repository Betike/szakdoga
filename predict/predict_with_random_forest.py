import pandas as pd
import os
import pickle
import json
import argparse
from datetime import datetime
import traceback

def load_model():
    model_path = "models/random_forest_prediction_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_random_forest_model.py first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    label_encoder_path = "utils/random_forest/random_forest_label_encoder.pkl"
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}. Please run train_random_forest_model.py first.")
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    feature_path = "utils/random_forest/random_forest_features.json"
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Features file not found at {feature_path}. Please run train_random_forest_model.py first.")
    
    with open(feature_path, 'r') as f:
        feature_info = json.load(f)
        feature_names = feature_info['feature_names']
    
    return model, label_encoder, feature_names

def prepare_match_data(home_team, away_team, team_stats, feature_names):
    if home_team == away_team:
        raise ValueError("Home team and away team cannot be identical")
    
    latest_season = team_stats['Season'].max()
    current_stats = team_stats[team_stats['Season'] == latest_season].copy()
    
    if home_team not in current_stats['Squad'].values:
        print(f"{home_team} not found in team stats. Current teams are: {current_stats['Squad'].unique()}")
        return None
    
    if away_team not in current_stats['Squad'].values:
        print(f"{away_team} not found in team stats. Current teams are: {current_stats['Squad'].unique()}")
        return None
    
    home_team_stats = current_stats[current_stats['Squad'] == home_team].iloc[0].to_dict()
    away_team_stats = current_stats[current_stats['Squad'] == away_team].iloc[0].to_dict()
    
    match_data = {}
    
    for key, value in home_team_stats.items():
        if key not in ['Squad', 'Season']:
            match_data[f'Home_{key}'] = value
    
    for key, value in away_team_stats.items():
        if key not in ['Squad', 'Season']:
            match_data[f'Away_{key}'] = value
    
    for key in home_team_stats.keys():
        if key not in ['Squad', 'Season']:
            home_val = home_team_stats[key]
            away_val = away_team_stats[key]
            if isinstance(home_val, (int, float)) and isinstance(away_val, (int, float)):
                match_data[f'Diff_{key}'] = home_val - away_val
    
    df = pd.DataFrame([match_data])
    
    missing_features = set(feature_names) - set(df.columns)
    extra_features = set(df.columns) - set(feature_names)
    
    for feature in missing_features:
        df[feature] = 0
    
    if extra_features:
        df = df.drop(columns=list(extra_features))
    
    df = df[feature_names]
    
    return df

def predict_match(model, label_encoder, home_team, away_team, team_stats, feature_names):
    if home_team == away_team:
        raise ValueError("Home team and away team cannot be identical")
    
    features = prepare_match_data(home_team, away_team, team_stats, feature_names)
    
    if features is None:
        return None, None
    
    predicted_proba = model.predict_proba(features)[0]
    predicted_class = model.predict(features)[0]
    
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    outcome_names = ['Away win', 'Draw', 'Home win']
    outcome_mapping = dict(zip(['A', 'D', 'H'], outcome_names))
    
    probability_dict = {}
    for idx, prob in enumerate(predicted_proba):
        class_label = label_encoder.inverse_transform([idx])[0]
        outcome_name = outcome_mapping.get(class_label, class_label)
        probability_dict[outcome_name] = float(prob)
    
    return predicted_label, probability_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Predict football match outcomes using Random Forest')
    parser.add_argument('--single-match', action='store_true', help='Predict a single match')
    parser.add_argument('--home', type=str, help='Home team name')
    parser.add_argument('--away', type=str, help='Away team name')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--file', type=str, help='File with matches to predict')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        
        if args.json:
            pass
        else:
            print("Loading Random Forest model...")
        
        model, label_encoder, feature_names = load_model()
        
        if not args.json:
            print(f"Model loaded successfully")
        
        team_stats_path = "data/processed/team_performance_dataset.csv"
        if not os.path.exists(team_stats_path):
            raise FileNotFoundError(f"Team statistics file not found at {team_stats_path}")
        
        team_stats = pd.read_csv(team_stats_path)
        
        latest_season = team_stats['Season'].max()
        if not args.json:
            print(f"Using statistics from season: {latest_season}")
        
        if args.single_match and args.home and args.away:
            home_team = args.home
            away_team = args.away
            
            try:
                result, probabilities = predict_match(model, label_encoder, home_team, away_team, team_stats, feature_names)
                
                if result is not None:
                    if args.json:
                        output = {
                            "prediction": result,
                            "probabilities": probabilities
                        }
                        print(json.dumps(output))
                    else:
                        print(f"\nPredicted result for {home_team} vs {away_team}: {result}")
                        
                        print("\nProbabilities:")
                        for outcome, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                            print(f"{outcome}: {prob:.2%}")
                else:
                    if args.json:
                        print(json.dumps({"error": "Could not make prediction with the given teams"}))
                    else:
                        print("Error: Could not make prediction with the given teams")
            except Exception as e:
                if args.json:
                    print(json.dumps({"error": str(e)}))
                else:
                    print(f"Error: {e}")
        
        elif args.file:
            file_path = args.file
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at {file_path}")
            
            matches = pd.read_csv(file_path)
            required_columns = ['HomeTeam', 'AwayTeam']
            
            if not all(col in matches.columns for col in required_columns):
                raise ValueError(f"Input file must contain columns: {required_columns}")
            
            identical_teams = matches[matches['HomeTeam'] == matches['AwayTeam']]
            if len(identical_teams) > 0 and not args.json:
                print(f"Found {len(identical_teams)} matches with identical home and away teams.")
                print("These matches will be skipped:")
                for idx, row in identical_teams.iterrows():
                    print(f"- {row['HomeTeam']} vs {row['AwayTeam']}")
                
                matches = matches[matches['HomeTeam'] != matches['AwayTeam']]
                print(f"Predicting for {len(matches)} valid matches.")
            
            matches['PredictedResult'] = None
            matches['HomeWinProb'] = None
            matches['DrawProb'] = None
            matches['AwayWinProb'] = None
            
            if not args.json:
                print(f"\nPredicting outcomes for {len(matches)} matches...")
            
            for idx, row in matches.iterrows():
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']
                
                try:
                    result, probabilities = predict_match(model, label_encoder, home_team, away_team, team_stats, feature_names)
                    
                    if result is not None:
                        matches.at[idx, 'PredictedResult'] = result
                        
                        matches.at[idx, 'HomeWinProb'] = probabilities.get('Home win', 0)
                        matches.at[idx, 'DrawProb'] = probabilities.get('Draw', 0)
                        matches.at[idx, 'AwayWinProb'] = probabilities.get('Away win', 0)
                except Exception as e:
                    if not args.json:
                        print(f"Error predicting match {home_team} vs {away_team}: {e}")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"random_forest_predictions_{timestamp}.csv"
            matches.to_csv(output_file, index=False)
            
            if not args.json:
                print(f"\nPredictions completed and saved to {output_file}")
                
                
                result_counts = matches['PredictedResult'].value_counts(normalize=True)
            else:
                print(json.dumps(matches.to_dict(orient='records')))
        
        else:
            if not args.json:
                choice = input("Single match (1) / Multiple matches from a file(2): ")
                
                if choice == '1':
                    print("\nEnter team names for prediction:")
                    home_team = input("Home team: ")
                    away_team = input("Away team: ")
                    
                    if home_team == away_team:
                        print("\nError: Home team and away team cannot be identical")
                    else:
                        try:
                            result, probabilities = predict_match(model, label_encoder, home_team, away_team, team_stats, feature_names)
                            
                            if result is not None:
                                print(f"\nPredicted result for {home_team} vs {away_team}: {result}")
                                
                                print("\nProbabilities:")
                                for outcome, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                                    print(f"{outcome}: {prob:.2%}")
                        except Exception as e:
                            print(f"\nError: {e}")
                
                elif choice == '2':
                    file_path = input("Enter path to CSV file containing matches (format: HomeTeam,AwayTeam): ")
                    
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"File not found at {file_path}")
                    
                    matches = pd.read_csv(file_path)
                    required_columns = ['HomeTeam', 'AwayTeam']
                    
                    if not all(col in matches.columns for col in required_columns):
                        raise ValueError(f"Input file must contain columns: {required_columns}")
                    
                    identical_teams = matches[matches['HomeTeam'] == matches['AwayTeam']]
                    if len(identical_teams) > 0:
                        print(f"Found {len(identical_teams)} matches with identical home and away teams.")
                        print("These matches will be skipped:")
                        for idx, row in identical_teams.iterrows():
                            print(f"- {row['HomeTeam']} vs {row['AwayTeam']}")
                        
                        matches = matches[matches['HomeTeam'] != matches['AwayTeam']]
                        print(f"Predicting for {len(matches)} valid matches.")
                    
                    matches['PredictedResult'] = None
                    matches['HomeWinProb'] = None
                    matches['DrawProb'] = None
                    matches['AwayWinProb'] = None
                    
                    print(f"\nPredicting outcomes for {len(matches)} matches...")
                    
                    for idx, row in matches.iterrows():
                        home_team = row['HomeTeam']
                        away_team = row['AwayTeam']
                        
                        try:
                            result, probabilities = predict_match(model, label_encoder, home_team, away_team, team_stats, feature_names)
                            
                            if result is not None:
                                matches.at[idx, 'PredictedResult'] = result
                                
                                matches.at[idx, 'HomeWinProb'] = probabilities.get('Home win', 0)
                                matches.at[idx, 'DrawProb'] = probabilities.get('Draw', 0)
                                matches.at[idx, 'AwayWinProb'] = probabilities.get('Away win', 0)
                        except Exception as e:
                            print(f"Error predicting match {home_team} vs {away_team}: {e}")
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"random_forest_predictions_{timestamp}.csv"
                    matches.to_csv(output_file, index=False)
                    
                    print(f"\nPredictions completed and saved to {output_file}")
                
                else:
                    print("Invalid choice. Please run again and select 1 or 2.")
            else:
                # JSON error for missing arguments
                print(json.dumps({"error": "Missing required arguments. Use --single-match with --home and --away, or use --file"}))
    
    except Exception as e:
        if args.json if 'args' in locals() else False:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}")
            traceback.print_exc() 
import pandas as pd
import numpy as np
import os
import pickle
import json
import argparse
from datetime import datetime

# Function to load the trained Random Forest model and label encoder
def load_model():
    # Load the model
    model_path = "models/random_forest_prediction_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_random_forest_model.py first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load label encoder
    label_encoder_path = "utils/random_forest/random_forest_label_encoder.pkl"
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}. Please run train_random_forest_model.py first.")
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load feature names
    feature_path = "utils/random_forest/random_forest_features.json"
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Features file not found at {feature_path}. Please run train_random_forest_model.py first.")
    
    with open(feature_path, 'r') as f:
        feature_info = json.load(f)
        feature_names = feature_info['feature_names']
    
    return model, label_encoder, feature_names

# Function to prepare data for a single match prediction
def prepare_match_data(home_team, away_team, team_stats, feature_names):
    """
    Prepare feature data for a single match prediction.
    
    Args:
        home_team (str): Name of the home team
        away_team (str): Name of the away team
        team_stats (pd.DataFrame): Team statistics dataframe
        feature_names (list): List of feature names the model expects
        
    Returns:
        pandas.DataFrame: DataFrame with features for prediction
    """
    # Check if home_team and away_team are identical
    if home_team == away_team:
        raise ValueError("Home team and away team cannot be identical. A team cannot play against itself.")
    
    # Filter for the most recent season
    latest_season = team_stats['Season'].max()
    current_stats = team_stats[team_stats['Season'] == latest_season].copy()
    
    # Check if teams exist in our dataset
    if home_team not in current_stats['Squad'].values:
        print(f"Warning: {home_team} not found in team stats. Predictions may be inaccurate.")
        return None
    
    if away_team not in current_stats['Squad'].values:
        print(f"Warning: {away_team} not found in team stats. Predictions may be inaccurate.")
        return None
    
    # Get team stats
    home_team_stats = current_stats[current_stats['Squad'] == home_team].iloc[0].to_dict()
    away_team_stats = current_stats[current_stats['Squad'] == away_team].iloc[0].to_dict()
    
    # Create a single row dataframe
    match_data = {}
    
    # Add home team features
    for key, value in home_team_stats.items():
        if key not in ['Squad', 'Season']:
            match_data[f'Home_{key}'] = value
    
    # Add away team features
    for key, value in away_team_stats.items():
        if key not in ['Squad', 'Season']:
            match_data[f'Away_{key}'] = value
    
    # Calculate difference features
    for key in home_team_stats.keys():
        if key not in ['Squad', 'Season']:
            home_val = home_team_stats[key]
            away_val = away_team_stats[key]
            if isinstance(home_val, (int, float)) and isinstance(away_val, (int, float)):
                match_data[f'Diff_{key}'] = home_val - away_val
    
    # Convert to DataFrame
    df = pd.DataFrame([match_data])
    
    # Align features with those used during training
    missing_features = set(feature_names) - set(df.columns)
    extra_features = set(df.columns) - set(feature_names)
    
    # Add missing features as 0
    for feature in missing_features:
        df[feature] = 0
    
    # Remove extra features
    if extra_features:
        df = df.drop(columns=list(extra_features))
    
    # Ensure correct column order
    df = df[feature_names]
    
    return df

# Function to predict match outcome using the Random Forest model
def predict_match(model, label_encoder, home_team, away_team, team_stats, feature_names):
    """
    Predict the outcome of a match between two teams.
    
    Args:
        model: Trained Random Forest model
        label_encoder: Label encoder for mapping predictions back to labels
        home_team (str): Name of the home team
        away_team (str): Name of the away team
        team_stats (pd.DataFrame): Team statistics dataframe
        feature_names (list): List of feature names the model expects
        
    Returns:
        tuple: (Predicted result, Probability of each outcome)
    """
    # Check if home_team and away_team are identical
    if home_team == away_team:
        raise ValueError("Home team and away team cannot be identical. A team cannot play against itself.")
    
    features = prepare_match_data(home_team, away_team, team_stats, feature_names)
    
    if features is None:
        return None, None
    
    # Make prediction
    predicted_proba = model.predict_proba(features)[0]
    predicted_class = model.predict(features)[0]
    
    # Map prediction back to label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    # Create a mapping for clearer output
    class_names = label_encoder.classes_
    outcome_names = ['Away win', 'Draw', 'Home win']
    outcome_mapping = dict(zip(['A', 'D', 'H'], outcome_names))
    
    # Map class indices to outcome names
    probability_dict = {}
    for idx, prob in enumerate(predicted_proba):
        class_label = label_encoder.inverse_transform([idx])[0]
        outcome_name = outcome_mapping.get(class_label, class_label)
        probability_dict[outcome_name] = float(prob)  # Convert numpy float to Python float for JSON
    
    return predicted_label, probability_dict

# Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Predict football match outcomes using Random Forest')
    parser.add_argument('--single-match', action='store_true', help='Predict a single match')
    parser.add_argument('--home', type=str, help='Home team name')
    parser.add_argument('--away', type=str, help='Away team name')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    parser.add_argument('--file', type=str, help='File with matches to predict')
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Load the model and label encoder
        if args.json:
            # Minimal output for JSON mode
            pass
        else:
            print("Loading Random Forest model...")
        
        model, label_encoder, feature_names = load_model()
        
        if not args.json:
            print(f"Model loaded successfully (trained with {len(feature_names)} features)")
        
        # Load team statistics
        team_stats_path = "data/processed/team_performance_dataset.csv"
        if not os.path.exists(team_stats_path):
            raise FileNotFoundError(f"Team statistics file not found at {team_stats_path}")
        
        team_stats = pd.read_csv(team_stats_path)
        
        # Display information about the model (only in non-JSON mode)
        latest_season = team_stats['Season'].max()
        if not args.json:
            print(f"Using team statistics from season: {latest_season}")
            print(f"Found stats for {len(team_stats[team_stats['Season'] == latest_season])} teams")
        
        # Process based on arguments
        if args.single_match and args.home and args.away:
            home_team = args.home
            away_team = args.away
            
            try:
                result, probabilities = predict_match(model, label_encoder, home_team, away_team, team_stats, feature_names)
                
                if result is not None:
                    if args.json:
                        # Output as JSON
                        output = {
                            "prediction": result,
                            "probabilities": probabilities
                        }
                        print(json.dumps(output))
                    else:
                        # Output for human reading
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
            # Process batch prediction file
            file_path = args.file
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at {file_path}")
            
            matches = pd.read_csv(file_path)
            required_columns = ['HomeTeam', 'AwayTeam']
            
            if not all(col in matches.columns for col in required_columns):
                raise ValueError(f"Input file must contain columns: {required_columns}")
            
            # Check for identical teams in the dataset
            identical_teams = matches[matches['HomeTeam'] == matches['AwayTeam']]
            if len(identical_teams) > 0 and not args.json:
                print(f"Warning: Found {len(identical_teams)} matches with identical home and away teams.")
                print("These matches will be skipped:")
                for idx, row in identical_teams.iterrows():
                    print(f"- {row['HomeTeam']} vs {row['AwayTeam']}")
                
                # Remove matches with identical teams
                matches = matches[matches['HomeTeam'] != matches['AwayTeam']]
                print(f"Proceeding with {len(matches)} valid matches.")
            
            # Add columns for predictions
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
                        
                        # Store probabilities
                        matches.at[idx, 'HomeWinProb'] = probabilities.get('Home win', 0)
                        matches.at[idx, 'DrawProb'] = probabilities.get('Draw', 0)
                        matches.at[idx, 'AwayWinProb'] = probabilities.get('Away win', 0)
                except Exception as e:
                    if not args.json:
                        print(f"Error predicting match {home_team} vs {away_team}: {e}")
            
            # Save predictions to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"random_forest_predictions_{timestamp}.csv"
            matches.to_csv(output_file, index=False)
            
            if not args.json:
                print(f"\nPredictions completed and saved to {output_file}")
                
                # Display predictions summary
                print("\nPredictions summary:")
                print(matches[['HomeTeam', 'AwayTeam', 'PredictedResult', 
                              'HomeWinProb', 'DrawProb', 'AwayWinProb']].head(10))
                
                # Summary statistics
                result_counts = matches['PredictedResult'].value_counts(normalize=True)
                print("\nOverall prediction distribution:")
                for result, pct in result_counts.items():
                    print(f"{result}: {pct:.2%}")
            else:
                # JSON output for batch predictions
                print(json.dumps(matches.to_dict(orient='records')))
        
        else:
            # Interactive mode - only if not using JSON output
            if not args.json:
                # Allow user to choose prediction method
                choice = input("Do you want to predict (1) a single match or (2) multiple matches from a file? (1/2): ")
                
                if choice == '1':
                    # Single match prediction
                    print("\nEnter team names for prediction:")
                    home_team = input("Home team: ")
                    away_team = input("Away team: ")
                    
                    # Check if the teams are identical
                    if home_team == away_team:
                        print("\nError: Home team and away team cannot be identical. A team cannot play against itself.")
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
                    # Batch prediction from file
                    file_path = input("Enter path to CSV file containing matches (format: HomeTeam,AwayTeam): ")
                    
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"File not found at {file_path}")
                    
                    matches = pd.read_csv(file_path)
                    required_columns = ['HomeTeam', 'AwayTeam']
                    
                    if not all(col in matches.columns for col in required_columns):
                        raise ValueError(f"Input file must contain columns: {required_columns}")
                    
                    # Check for identical teams in the dataset
                    identical_teams = matches[matches['HomeTeam'] == matches['AwayTeam']]
                    if len(identical_teams) > 0:
                        print(f"Warning: Found {len(identical_teams)} matches with identical home and away teams.")
                        print("These matches will be skipped:")
                        for idx, row in identical_teams.iterrows():
                            print(f"- {row['HomeTeam']} vs {row['AwayTeam']}")
                        
                        # Remove matches with identical teams
                        matches = matches[matches['HomeTeam'] != matches['AwayTeam']]
                        print(f"Proceeding with {len(matches)} valid matches.")
                    
                    # Add columns for predictions
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
                                
                                # Store probabilities
                                matches.at[idx, 'HomeWinProb'] = probabilities.get('Home win', 0)
                                matches.at[idx, 'DrawProb'] = probabilities.get('Draw', 0)
                                matches.at[idx, 'AwayWinProb'] = probabilities.get('Away win', 0)
                        except Exception as e:
                            print(f"Error predicting match {home_team} vs {away_team}: {e}")
                    
                    # Save predictions to file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"random_forest_predictions_{timestamp}.csv"
                    matches.to_csv(output_file, index=False)
                    
                    print(f"\nPredictions completed and saved to {output_file}")
                    
                    # Display predictions summary
                    print("\nPredictions summary:")
                    print(matches[['HomeTeam', 'AwayTeam', 'PredictedResult', 
                                'HomeWinProb', 'DrawProb', 'AwayWinProb']].head(10))
                    
                    # Summary statistics
                    result_counts = matches['PredictedResult'].value_counts(normalize=True)
                    print("\nOverall prediction distribution:")
                    for result, pct in result_counts.items():
                        print(f"{result}: {pct:.2%}")
                
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
            import traceback
            traceback.print_exc() 
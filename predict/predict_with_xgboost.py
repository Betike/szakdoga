import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
import json
import argparse
from datetime import datetime

# Function to load the trained XGBoost model and label encoder
def load_model():
    # Load the model
    model_path = "models/xgboost_prediction_model.json"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_xgboost_model.py first.")
    
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Load label encoder
    label_encoder_path = "utils/xgboost/xgboost_label_encoder.pkl"
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Label encoder file not found at {label_encoder_path}. Please run train_xgboost_model.py first.")
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder

# Function to prepare data for a single match prediction
def prepare_match_data(home_team, away_team, team_stats):
    """
    Prepare feature data for a single match prediction.
    
    Args:
        home_team (str): Name of the home team
        away_team (str): Name of the away team
        team_stats (pd.DataFrame): Team statistics dataframe
        
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
    
    return df

# Load the feature columns used during training
def get_training_features(model):
    """Get feature names used during training of the XGBoost model"""
    feature_names = model.feature_names
    if feature_names is None:
        # If feature names are not stored in the model, load them from feature importance file
        try:
            feature_imp_path = "utils/xgboost/xgboost_feature_importance.csv"
            if os.path.exists(feature_imp_path):
                feature_imp = pd.read_csv(feature_imp_path)
                return feature_imp['Feature'].tolist()
            else:
                raise FileNotFoundError(f"Feature importance file not found at {feature_imp_path}")
        except Exception as e:
            print(f"Error loading feature names: {e}")
            return None
    return feature_names

# Function to predict match outcome using the XGBoost model
def predict_match(model, label_encoder, home_team, away_team, team_stats):
    """
    Predict the outcome of a match between two teams.
    
    Args:
        model (xgb.Booster): Trained XGBoost model
        label_encoder: Label encoder for mapping predictions back to labels
        home_team (str): Name of the home team
        away_team (str): Name of the away team
        team_stats (pd.DataFrame): Team statistics dataframe
        
    Returns:
        tuple: (Predicted result, Probability of each outcome)
    """
    # Check if home_team and away_team are identical
    if home_team == away_team:
        raise ValueError("Home team and away team cannot be identical. A team cannot play against itself.")
    
    features = prepare_match_data(home_team, away_team, team_stats)
    
    if features is None:
        return None, None
    
    # Get training features
    training_features = get_training_features(model)
    
    if training_features:
        # Align features with those used during training
        for feature in training_features:
            if feature not in features.columns:
                # Add missing features with default value of 0
                features[feature] = 0
        
        # Remove extra features not used in training
        extra_features = [col for col in features.columns if col not in training_features]
        for feature in extra_features:
            features = features.drop(columns=[feature])
        
        # Ensure the same column order as during training
        features = features[training_features]
    else:
        print("Warning: Could not determine training features. Predictions may fail.")
    
    # Convert to DMatrix (required by XGBoost)
    dmatrix = xgb.DMatrix(features)
    
    # Make prediction
    probabilities = model.predict(dmatrix)[0]
    predicted_idx = np.argmax(probabilities)
    
    # Map prediction back to label
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    
    # Create a mapping for clearer output
    outcome_names = ['Away win', 'Draw', 'Home win']
    probability_dict = dict(zip(outcome_names, probabilities.tolist()))
    
    return predicted_label, probability_dict

# Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Predict football match outcomes using XGBoost')
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
            print("Loading XGBoost model...")
        
        model, label_encoder = load_model()
        
        # Print model feature information (only in non-JSON mode)
        if not args.json:
            training_features = get_training_features(model)
            if training_features:
                print(f"Model trained with {len(training_features)} features")
        
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
                result, probabilities = predict_match(model, label_encoder, home_team, away_team, team_stats)
                
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
                    result, probabilities = predict_match(model, label_encoder, home_team, away_team, team_stats)
                    
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
            output_file = f"xgboost_predictions_{timestamp}.csv"
            matches.to_csv(output_file, index=False)
            
            if not args.json:
                print(f"\nPredictions completed and saved to {output_file}")
                
                # Display predictions summary
                print("\nPredictions summary:")
                print(matches[['HomeTeam', 'AwayTeam', 'PredictedResult', 
                              'HomeWinProb', 'DrawProb', 'AwayWinProb']].head(10))
                
                # Calculate team-specific prediction breakdown
                team_predictions = {}
                for team in pd.concat([matches['HomeTeam'], matches['AwayTeam']]).unique():
                    team_matches = matches[(matches['HomeTeam'] == team) | (matches['AwayTeam'] == team)]
                    team_predictions[team] = team_matches['PredictedResult'].value_counts(normalize=True).to_dict()
                
                # Save team predictions to file
                team_preds_df = pd.DataFrame(team_predictions).T
                team_preds_df.fillna(0, inplace=True)
                team_preds_file = f"data_test/xgboost_team_predictions_{timestamp}.csv"
                team_preds_df.to_csv(team_preds_file)
                print(f"\nTeam-specific prediction breakdown saved to {team_preds_file}")
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
                            result, probabilities = predict_match(model, label_encoder, home_team, away_team, team_stats)
                            
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
                            result, probabilities = predict_match(model, label_encoder, home_team, away_team, team_stats)
                            
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
                    output_file = f"xgboost_predictions_{timestamp}.csv"
                    matches.to_csv(output_file, index=False)
                    
                    print(f"\nPredictions completed and saved to {output_file}")
                    
                    # Display predictions summary
                    print("\nPredictions summary:")
                    print(matches[['HomeTeam', 'AwayTeam', 'PredictedResult', 
                                  'HomeWinProb', 'DrawProb', 'AwayWinProb']].head(10))
                    
                    # Calculate team-specific prediction breakdown
                    team_predictions = {}
                    for team in pd.concat([matches['HomeTeam'], matches['AwayTeam']]).unique():
                        team_matches = matches[(matches['HomeTeam'] == team) | (matches['AwayTeam'] == team)]
                        team_predictions[team] = team_matches['PredictedResult'].value_counts(normalize=True).to_dict()
                    
                    # Save team predictions to file
                    team_preds_df = pd.DataFrame(team_predictions).T
                    team_preds_df.fillna(0, inplace=True)
                    team_preds_file = f"data_test/xgboost_team_predictions_{timestamp}.csv"
                    team_preds_df.to_csv(team_preds_file)
                    print(f"\nTeam-specific prediction breakdown saved to {team_preds_file}")
                
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
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import json
import sys
import argparse
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Neural Network Model (same architecture as in training)
class MatchPredictionNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.4):
        super(MatchPredictionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 3)  # 3 classes: Away win, Draw, Home win
        )
    
    def forward(self, x):
        return self.model(x)

# Function to load the model
def load_model(model_path, feature_info_path):
    # Load feature scaling information
    with open(feature_info_path, 'r') as f:
        feature_info = json.load(f)
    
    # Create scaler with saved parameters
    scaler = StandardScaler()
    
    # Ensure proper numpy arrays for scaler parameters
    if isinstance(feature_info['scaler_mean'], list):
        scaler.mean_ = np.array(feature_info['scaler_mean'])
    else:
        scaler.mean_ = feature_info['scaler_mean']
        
    if isinstance(feature_info['scaler_scale'], list):
        scaler.scale_ = np.array(feature_info['scaler_scale'])
    else:
        scaler.scale_ = feature_info['scaler_scale']
    
    # Validate scaler
    if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
        raise ValueError("Invalid scaler parameters in feature info file")
    
    # Create model with correct input size
    input_size = len(feature_info['feature_names'])
    model = MatchPredictionNN(input_size)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    return model, scaler, feature_info['feature_names']

# Function to prepare data for a single match prediction
def prepare_match_data(home_team, away_team, team_stats, feature_names, scaler):
    """
    Prepare feature data for a single match prediction.
    
    Args:
        home_team (str): Name of the home team
        away_team (str): Name of the away team
        team_stats (pd.DataFrame): Team statistics dataframe
        feature_names (list): List of feature names the model expects
        scaler (StandardScaler): Fitted scaler for feature normalization
        
    Returns:
        torch.Tensor: Tensor with features for prediction
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
    
    # Convert to DataFrame with only the expected features
    df = pd.DataFrame([match_data])
    
    # Add missing features with 0 values
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Keep only the features that the model expects, in the same order
    df = df[feature_names]
    
    # Convert DataFrame to numpy array to avoid feature name warning
    features_array = df.values
    
    # Scale features
    features_scaled = scaler.transform(features_array)
    features_tensor = torch.FloatTensor(features_scaled)
    
    return features_tensor

# Function to predict match outcome using the PyTorch model
def predict_match(model, home_team, away_team, team_stats, feature_names, scaler):
    """
    Predict the outcome of a match between two teams.
    
    Args:
        model (MatchPredictionNN): Trained PyTorch model
        home_team (str): Name of the home team
        away_team (str): Name of the away team
        team_stats (pd.DataFrame): Team statistics dataframe
        feature_names (list): List of feature names the model expects
        scaler (StandardScaler): Fitted scaler for feature normalization
        
    Returns:
        tuple: (Predicted result, Probability of each outcome)
    """
    # Check if home_team and away_team are identical
    if home_team == away_team:
        raise ValueError("Home team and away team cannot be identical. A team cannot play against itself.")
    
    features = prepare_match_data(home_team, away_team, team_stats, feature_names, scaler)
    
    if features is None:
        return None, None
    
    # Make prediction
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
    
    # Convert to numpy
    result_class = predicted.item()
    probabilities = probabilities.cpu().numpy()
    
    # Map to outcome labels
    label_map_inverse = {0: 'A', 1: 'D', 2: 'H'}
    result = label_map_inverse[result_class]
    
    # Convert to dictionary format for JSON serialization
    outcome_names = ['Away win', 'Draw', 'Home win']
    probability_dict = {name: float(prob) for name, prob in zip(outcome_names, probabilities)}
    
    return result, probability_dict

# Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Predict football match outcomes using PyTorch neural network')
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
        
        # Load the model and required data
        model_path = "models/pytorch_model.pth"
        feature_info_path = "utils/neural_network/feature_scaling_info.json"
        team_stats_path = "data/processed/team_performance_dataset.csv"
        
        if not os.path.exists(model_path):
            error_msg = f"Error: Model file not found at {model_path}"
            if args.json:
                print(json.dumps({"error": error_msg}))
                exit(1)
            else:
                print(error_msg)
                print("Please run train_pytorch_model.py first.")
                exit(1)
    
        if not os.path.exists(feature_info_path):
            error_msg = f"Error: Feature info file not found at {feature_info_path}"
            if args.json:
                print(json.dumps({"error": error_msg}))
                exit(1)
            else:
                print(error_msg)
                print("Please run train_pytorch_model.py first.")
                exit(1)
    
        if not args.json:
            print("Loading PyTorch model and team statistics...")
            
        try:
            model, scaler, feature_names = load_model(model_path, feature_info_path)
            team_stats = pd.read_csv(team_stats_path)
        except Exception as e:
            error_msg = f"Error loading model or team statistics: {e}"
            if args.json:
                print(json.dumps({"error": error_msg}))
                exit(1)
            else:
                print(error_msg)
                exit(1)
    
        # Display information about the model
        latest_season = team_stats['Season'].max()
        if not args.json:
            print(f"Using team statistics from season: {latest_season}")
            print(f"Found stats for {len(team_stats[team_stats['Season'] == latest_season])} teams")
    
        # Process based on arguments
        if args.single_match and args.home and args.away:
            home_team = args.home
            away_team = args.away
            
            try:
                result, probabilities = predict_match(model, home_team, away_team, team_stats, feature_names, scaler)
                
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
                        outcome_names = ['Away win', 'Draw', 'Home win']
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
                error_msg = f"Error: File not found at {file_path}"
                if args.json:
                    print(json.dumps({"error": error_msg}))
                    exit(1)
                else:
                    print(error_msg)
                    exit(1)
            
            try:
                matches = pd.read_csv(file_path)
                required_columns = ['HomeTeam', 'AwayTeam']
                
                if not all(col in matches.columns for col in required_columns):
                    error_msg = f"Error: Input file must contain columns: {required_columns}"
                    if args.json:
                        print(json.dumps({"error": error_msg}))
                        exit(1)
                    else:
                        print(error_msg)
                        exit(1)
                
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
                        result, probabilities = predict_match(model, home_team, away_team, team_stats, feature_names, scaler)
                        
                        if result is not None:
                            matches.at[idx, 'PredictedResult'] = result
                            
                            # Store probabilities
                            matches.at[idx, 'AwayWinProb'] = probabilities['Away win']
                            matches.at[idx, 'DrawProb'] = probabilities['Draw']
                            matches.at[idx, 'HomeWinProb'] = probabilities['Home win']
                    except Exception as e:
                        if not args.json:
                            print(f"Error predicting match {home_team} vs {away_team}: {e}")
                
                # Save predictions to file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"pytorch_predictions_{timestamp}.csv"
                matches.to_csv(output_file, index=False)
                
                if not args.json:
                    print(f"\nPredictions completed and saved to {output_file}")
                    
                    # Display predictions summary
                    print("\nPredictions summary:")
                    print(matches[['HomeTeam', 'AwayTeam', 'PredictedResult', 
                                'HomeWinProb', 'DrawProb', 'AwayWinProb']].head(10))
                    
                    # Calculate team-specific accuracy
                    team_accuracy = {}
                    for team in pd.concat([matches['HomeTeam'], matches['AwayTeam']]).unique():
                        team_matches = matches[(matches['HomeTeam'] == team) | (matches['AwayTeam'] == team)]
                        team_accuracy[team] = team_matches['PredictedResult'].value_counts(normalize=True).to_dict()
                    
                    # Save team accuracy to file
                    team_accuracy_df = pd.DataFrame(team_accuracy).T
                    team_accuracy_df.fillna(0, inplace=True)
                    team_accuracy_file = f"utils/neural_network/team_prediction_accuracy_{timestamp}.csv"
                    team_accuracy_df.to_csv(team_accuracy_file)
                    print(f"\nTeam-specific prediction breakdown saved to {team_accuracy_file}")
                else:
                    # JSON output for batch predictions
                    print(json.dumps(matches.to_dict(orient='records')))
                
            except Exception as e:
                if args.json:
                    print(json.dumps({"error": str(e)}))
                else:
                    print(f"Error processing file: {e}")
        
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
                            result, probabilities = predict_match(model, home_team, away_team, team_stats, feature_names, scaler)
                            
                            if result is not None:
                                print(f"\nPredicted result for {home_team} vs {away_team}: {result}")
                                
                                # Map the probabilities to outcomes
                                outcome_names = ['Away win', 'Draw', 'Home win']
                                print("\nProbabilities:")
                                for outcome, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                                    print(f"{outcome}: {prob:.2%}")
                        except Exception as e:
                            print(f"\nError: {e}")
                
                elif choice == '2':
                    # Batch prediction from file
                    file_path = input("Enter path to CSV file containing matches (format: HomeTeam,AwayTeam): ")
                    
                    if not os.path.exists(file_path):
                        print(f"Error: File not found at {file_path}")
                        exit(1)
                    
                    try:
                        matches = pd.read_csv(file_path)
                        required_columns = ['HomeTeam', 'AwayTeam']
                        
                        if not all(col in matches.columns for col in required_columns):
                            print(f"Error: Input file must contain columns: {required_columns}")
                            exit(1)
                        
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
                                result, probabilities = predict_match(model, home_team, away_team, team_stats, feature_names, scaler)
                                
                                if result is not None:
                                    matches.at[idx, 'PredictedResult'] = result
                                    
                                    # Store probabilities
                                    matches.at[idx, 'AwayWinProb'] = probabilities['Away win']
                                    matches.at[idx, 'DrawProb'] = probabilities['Draw']
                                    matches.at[idx, 'HomeWinProb'] = probabilities['Home win']
                            except Exception as e:
                                print(f"Error predicting match {home_team} vs {away_team}: {e}")
                        
                        # Save predictions to file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file = f"pytorch_predictions_{timestamp}.csv"
                        matches.to_csv(output_file, index=False)
                        
                        print(f"\nPredictions completed and saved to {output_file}")
                        
                        # Display predictions summary
                        print("\nPredictions summary:")
                        print(matches[['HomeTeam', 'AwayTeam', 'PredictedResult', 
                                      'HomeWinProb', 'DrawProb', 'AwayWinProb']].head(10))
                        
                        # Calculate team-specific accuracy
                        team_accuracy = {}
                        for team in pd.concat([matches['HomeTeam'], matches['AwayTeam']]).unique():
                            team_matches = matches[(matches['HomeTeam'] == team) | (matches['AwayTeam'] == team)]
                            team_accuracy[team] = team_matches['PredictedResult'].value_counts(normalize=True).to_dict()
                        
                        # Save team accuracy to file
                        team_accuracy_df = pd.DataFrame(team_accuracy).T
                        team_accuracy_df.fillna(0, inplace=True)
                        team_accuracy_file = f"utils/neural_network/team_prediction_accuracy_{timestamp}.csv"
                        team_accuracy_df.to_csv(team_accuracy_file)
                        print(f"\nTeam-specific prediction breakdown saved to {team_accuracy_file}")
                        
                    except Exception as e:
                        print(f"Error processing file: {e}")
                
                else:
                    print("Invalid choice. Please run again and select 1 or 2.")
            else:
                # JSON error for missing arguments
                print(json.dumps({"error": "Missing required arguments. Use --single-match with --home and --away, or use --file"}))
    
    except Exception as e:
        if 'args' in locals() and args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error in main function: {e}")
            print("\nDebug information:")
            print(f"Python version: {sys.version}")
            print(f"PyTorch version: {torch.__version__}")
            print(f"NumPy version: {np.__version__}")
            print(f"Pandas version: {pd.__version__}")
            
            # Print traceback for detailed error information
            import traceback
            print("\nError details:")
            traceback.print_exc() 
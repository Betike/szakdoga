import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_team_data import load_team_data, prepare_data_for_training, get_team_form
from models.team_based_predictor import TeamBasedPredictor

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

def get_team_name(conn, team_api_id):
    """Get team name from team_api_id"""
    query = "SELECT team_long_name FROM Team WHERE team_api_id = ?"
    result = conn.execute(query, (team_api_id,)).fetchone()
    return result[0] if result else f"Unknown Team ({team_api_id})"

def get_match_details(match_id, conn):
    """Get detailed information about a specific match"""
    query = """
    SELECT 
        m.match_api_id, m.date, 
        m.home_team_api_id, m.away_team_api_id,
        m.home_team_goal, m.away_team_goal,
        m.league_id, l.name as league_name
    FROM Match m
    JOIN League l ON m.league_id = l.id
    WHERE m.match_api_id = ?
    """
    
    match_info = conn.execute(query, (match_id,)).fetchone()
    
    if not match_info:
        return None
    
    # Convert to dictionary
    match_dict = {
        'match_id': match_info[0],
        'date': match_info[1],
        'home_team_id': match_info[2],
        'away_team_id': match_info[3],
        'home_goals': match_info[4],
        'away_goals': match_info[5],
        'league_id': match_info[6],
        'league_name': match_info[7],
        'home_team_name': get_team_name(conn, match_info[2]),
        'away_team_name': get_team_name(conn, match_info[3])
    }
    
    # Determine actual outcome (0: away win, 1: draw, 2: home win)
    if match_dict['home_goals'] > match_dict['away_goals']:
        match_dict['actual_outcome'] = 2
        match_dict['actual_outcome_str'] = 'Home Win'
    elif match_dict['home_goals'] < match_dict['away_goals']:
        match_dict['actual_outcome'] = 0
        match_dict['actual_outcome_str'] = 'Away Win'
    else:
        match_dict['actual_outcome'] = 1
        match_dict['actual_outcome_str'] = 'Draw'
    
    return match_dict

def get_team_attributes(conn, team_id, match_date):
    """Get team's tactical attributes before the match"""
    query = """
    SELECT 
        buildUpPlaySpeed, buildUpPlayPassing, 
        chanceCreationPassing, chanceCreationCrossing, chanceCreationShooting, 
        defencePressure, defenceAggression, defenceTeamWidth
    FROM Team_Attributes 
    WHERE team_api_id = ? AND date < ?
    ORDER BY date DESC
    LIMIT 1
    """
    
    result = conn.execute(query, (team_id, match_date)).fetchone()
    
    if not result:
        return pd.Series([50, 50, 50, 50, 50, 50, 50, 50], 
                        index=['buildup_speed', 'buildup_passing', 'chance_creation_passing', 
                              'chance_creation_crossing', 'chance_creation_shooting', 
                              'defence_pressure', 'defence_aggression', 'defence_width'])
    
    return pd.Series([
        result[0], result[1], 
        result[2], result[3], result[4], 
        result[5], result[6], result[7]
    ], index=['buildup_speed', 'buildup_passing', 'chance_creation_passing', 
              'chance_creation_crossing', 'chance_creation_shooting', 
              'defence_pressure', 'defence_aggression', 'defence_width'])

def prepare_match_features(match_info, conn):
    """Prepare features for a specific match to use with the model"""
    match_date = match_info['date']
    home_team_id = match_info['home_team_id']
    away_team_id = match_info['away_team_id']
    
    # Get team attributes
    home_attrs = get_team_attributes(conn, home_team_id, match_date)
    away_attrs = get_team_attributes(conn, away_team_id, match_date)
    
    # Get team form
    home_form = get_team_form(conn, home_team_id, match_date)
    away_form = get_team_form(conn, away_team_id, match_date)
    
    # Prepare feature vector (match the order used during training)
    features = pd.Series([
        # Home team tactical attributes
        home_attrs['buildup_speed'],
        home_attrs['buildup_passing'],
        (home_attrs['chance_creation_passing'] + 
         home_attrs['chance_creation_crossing'] + 
         home_attrs['chance_creation_shooting']) / 3,  # Average of chance creation attributes
        home_attrs['defence_pressure'],
        home_attrs['defence_aggression'],
        home_attrs['defence_width'],
        
        # Away team tactical attributes
        away_attrs['buildup_speed'],
        away_attrs['buildup_passing'],
        (away_attrs['chance_creation_passing'] + 
         away_attrs['chance_creation_crossing'] + 
         away_attrs['chance_creation_shooting']) / 3,  # Average of chance creation attributes
        away_attrs['defence_pressure'],
        away_attrs['defence_aggression'],
        away_attrs['defence_width']
    ])
    
    # Add form features
    features = pd.concat([features, home_form, away_form])
    
    return features

def get_layer_activations(model, x, layer_names):
    """Get activations from intermediate layers of the model"""
    model.eval()
    activations = {}
    
    # Home and away attribute encodings
    home_attr = x[:, :6]
    home_form = x[:, 6:16]
    away_attr = x[:, 16:22]
    away_form = x[:, 22:32]
    
    # Get team attribute encodings
    home_attr_encoded = model.team_attr_encoder(home_attr)
    home_form_encoded = model.form_encoder(home_form)
    away_attr_encoded = model.team_attr_encoder(away_attr)
    away_form_encoded = model.form_encoder(away_form)
    
    # Store activations
    activations['home_attr_encoded'] = home_attr_encoded.detach().numpy()
    activations['home_form_encoded'] = home_form_encoded.detach().numpy()
    activations['away_attr_encoded'] = away_attr_encoded.detach().numpy()
    activations['away_form_encoded'] = away_form_encoded.detach().numpy()
    
    # Get combined features
    combined = torch.cat([
        home_attr_encoded, home_form_encoded,
        away_attr_encoded, away_form_encoded
    ], dim=1)
    
    activations['combined'] = combined.detach().numpy()
    
    # Get final output before softmax
    output = model.combined_encoder(combined)
    activations['output'] = output.detach().numpy()
    
    # Calculate softmax probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)
    activations['probabilities'] = probabilities.detach().numpy()
    
    return activations

def visualize_match_prediction(match_id, conn, model, scaler):
    """Visualize the prediction process for a specific match"""
    # Get match details
    match_info = get_match_details(match_id, conn)
    if not match_info:
        print(f"Match with ID {match_id} not found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Match Analysis: {match_info['home_team_name']} vs {match_info['away_team_name']}")
    print(f"Date: {match_info['date']}")
    print(f"League: {match_info['league_name']}")
    print(f"Actual Result: {match_info['home_goals']} - {match_info['away_goals']} ({match_info['actual_outcome_str']})")
    print(f"{'='*80}\n")
    
    # Get features for this match
    features = prepare_match_features(match_info, conn)
    
    # Scale features to match training data
    scaled_features = pd.Series(scaler.transform([features.values])[0], index=features.index)
    
    # Prepare input tensor for the model
    input_tensor = torch.FloatTensor(scaled_features.values).unsqueeze(0)
    
    # Get layer activations
    activations = get_layer_activations(model, input_tensor, ['team_attr_encoder', 'form_encoder', 'combined_encoder'])
    
    # Get predicted probabilities and outcome
    probs = activations['probabilities'][0]
    predicted_class = np.argmax(probs)
    outcome_names = ['Away Win', 'Draw', 'Home Win']
    predicted_outcome = outcome_names[predicted_class]
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Match Information
    plt.subplot(2, 2, 1)
    plt.axis('off')
    match_text = (
        f"{match_info['home_team_name']} vs {match_info['away_team_name']}\n"
        f"Date: {match_info['date']}\n"
        f"League: {match_info['league_name']}\n\n"
        f"Actual Result: {match_info['home_goals']} - {match_info['away_goals']}\n"
        f"Actual Outcome: {match_info['actual_outcome_str']}\n\n"
        f"Predicted Outcome: {predicted_outcome}\n"
        f"Prediction Confidence: {probs[predicted_class]:.2f} ({probs[predicted_class]*100:.1f}%)\n\n"
        f"Away Win Probability: {probs[0]:.2f} ({probs[0]*100:.1f}%)\n"
        f"Draw Probability: {probs[1]:.2f} ({probs[1]*100:.1f}%)\n"
        f"Home Win Probability: {probs[2]:.2f} ({probs[2]*100:.1f}%)"
    )
    plt.text(0.1, 0.5, match_text, fontsize=14, verticalalignment='center')
    plt.title("Match Information", fontsize=16)
    
    # 2. Team Attributes Comparison
    plt.subplot(2, 2, 2)
    attr_labels = ['Buildup Speed', 'Buildup Passing', 'Chance Creation', 
                   'Defence Pressure', 'Defence Aggression', 'Defence Width']
    
    home_attrs = features.iloc[:6].values
    away_attrs = features.iloc[6:12].values
    
    x = np.arange(len(attr_labels))
    width = 0.35
    
    plt.bar(x - width/2, home_attrs, width, label=match_info['home_team_name'])
    plt.bar(x + width/2, away_attrs, width, label=match_info['away_team_name'])
    
    plt.ylabel('Rating')
    plt.title('Team Tactical Attributes Comparison')
    plt.xticks(x, attr_labels, rotation=30, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # 3. Team Form Comparison
    plt.subplot(2, 2, 3)
    form_labels = ['Goal Diff (Mean)', 'Goal Diff (Std)', 'Goals Scored (Mean)', 
                   'Goals Scored (Std)', 'Goals Conceded (Mean)', 'Goals Conceded (Std)', 
                   'Win Rate', 'Draw Rate', 'Loss Rate', 'Match Coverage']
    
    home_form = features.iloc[12:22].values
    away_form = features.iloc[22:32].values
    
    x = np.arange(len(form_labels))
    
    plt.bar(x - width/2, home_form, width, label=match_info['home_team_name'])
    plt.bar(x + width/2, away_form, width, label=match_info['away_team_name'])
    
    plt.ylabel('Value')
    plt.title('Team Recent Form Comparison')
    plt.xticks(x, form_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # 4. Prediction Probability
    plt.subplot(2, 2, 4)
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    # Check if prediction was correct
    prediction_correct = predicted_class == match_info['actual_outcome']
    border_color = 'green' if prediction_correct else 'red'
    edgecolor = [border_color if i == predicted_class else 'none' for i in range(3)]
    linewidth = [2 if i == predicted_class else 0 for i in range(3)]
    
    bars = plt.bar(outcome_names, probs, color=colors, edgecolor=edgecolor, linewidth=linewidth)
    
    # Add a marker for the actual outcome
    plt.axvline(x=outcome_names[match_info['actual_outcome']], color='black', linestyle='--', alpha=0.7)
    plt.text(outcome_names[match_info['actual_outcome']], 0.05, 'Actual Outcome', rotation=90, 
             verticalalignment='bottom', horizontalalignment='center')
    
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(f"visualizations/match_{match_id}_prediction.png")
    plt.close()
    
    print(f"Visualization saved to visualizations/match_{match_id}_prediction.png")
    
    # Return prediction results
    return {
        'match_info': match_info,
        'predicted_outcome': predicted_outcome,
        'probabilities': probs,
        'prediction_correct': prediction_correct
    }

def main():
    """Main function to visualize predictions for specific matches"""
    # Load the trained model
    model_path = 'models/team_based_predictor.pth'
    model = TeamBasedPredictor.load_model(model_path)
    model.eval()
    
    # Load and prepare data to get the scaler
    print("Loading data to get the scaler...")
    X, y, feature_columns = load_team_data()
    _, _, _, _, scaler = prepare_data_for_training(X, y)
    
    # Connect to the database
    conn = sqlite3.connect('database.sqlite')
    
    # Get some match IDs to visualize (alternatively, you could specify them directly)
    # Let's get some matches from different leagues and with different outcomes
    query = """
    SELECT match_api_id 
    FROM Match 
    WHERE season = '2015/2016' 
    ORDER BY RANDOM() 
    LIMIT 10
    """
    
    match_ids = [row[0] for row in conn.execute(query).fetchall()]
    
    # Visualize predictions for these matches
    correct_predictions = 0
    for match_id in match_ids:
        result = visualize_match_prediction(match_id, conn, model, scaler)
        if result and result['prediction_correct']:
            correct_predictions += 1
    
    print(f"\nAccuracy on selected matches: {correct_predictions/len(match_ids):.2f}")
    
    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main() 
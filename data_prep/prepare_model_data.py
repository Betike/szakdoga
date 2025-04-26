import pandas as pd
import numpy as np
import os
from datetime import datetime

# 1. Load the datasets
print("Loading datasets...")
match_data = pd.read_csv("data/raw/premier_league_match_outcomes_2017-2018_to_2024-2025.csv")
team_stats = pd.read_csv("data/processed/team_performance_dataset.csv")

# Display basic info
print(f"Match data shape: {match_data.shape}")
print(f"Team stats shape: {team_stats.shape}")

# 2. Clean and prepare the data
# Ensure date is in datetime format
match_data['Date'] = pd.to_datetime(match_data['Date'])

# Sort matches chronologically
match_data = match_data.sort_values('Date')

# Check for missing values
print("\nMissing values in match data:")
print(match_data.isnull().sum())

# Create a team-venue mapping using the match data
print("\nCreating team-venue mapping...")
team_venue_mapping = {}
for _, row in match_data.iterrows():
    team = row['HomeTeam']
    venue = row['Venue']
    if team not in team_venue_mapping and not pd.isna(venue):
        team_venue_mapping[team] = venue

print(f"Found {len(team_venue_mapping)} team-venue pairs")

# 3. Merge team stats onto each match
# First for home team
print("\nMerging home team statistics...")
home_stats = team_stats.rename(columns={col: f'Home_{col}' for col in team_stats.columns if col != 'Squad' and col != 'Season'})
home_stats = home_stats.rename(columns={'Squad': 'HomeTeam'})

model_data = pd.merge(
    match_data,
    home_stats,
    on=['HomeTeam', 'Season'],
    how='left'
)

# Then for away team
print("Merging away team statistics...")
away_stats = team_stats.rename(columns={col: f'Away_{col}' for col in team_stats.columns if col != 'Squad' and col != 'Season'})
away_stats = away_stats.rename(columns={'Squad': 'AwayTeam'})

model_data = pd.merge(
    model_data,
    away_stats,
    on=['AwayTeam', 'Season'],
    how='left'
)

# 4. Create difference features
print("\nCreating difference features...")
# Identify numeric columns from team stats (excluding 'Squad' and 'Season')
numeric_cols = team_stats.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col not in ['Squad', 'Season']:
        home_col = f'Home_{col}'
        away_col = f'Away_{col}'
        if home_col in model_data.columns and away_col in model_data.columns:
            model_data[f'Diff_{col}'] = model_data[home_col] - model_data[away_col]

# 5. Define target variable
print("\nDefining target variable...")
# Result column: 'H' for home win, 'A' for away win, 'D' for draw
print("Target variable (Result) distribution:")
print(model_data['Result'].value_counts())

# 6. Split data chronologically
print("\nSplitting data chronologically...")
# Determine cutoff date for train/test split (e.g., use 2023-2024 season as test)

holdout_seasons = ['2023-2024', '2024-2025']
train_data = model_data[~model_data['Season'].isin(holdout_seasons)].copy()
test_data = model_data[model_data['Season'].isin(holdout_seasons)].copy()

# Create a column with team-venue pairs
print("\nAdding team-venue pairs to datasets...")
def add_team_venue_pair(row):
    team = row['HomeTeam']
    venue = row['Venue']
    if pd.isna(venue) and team in team_venue_mapping:
        return f"{team}"
    elif not pd.isna(venue):
        return f"{team}"
    else:
        return f"{team} - Unknown Venue"

train_data['Venue'] = train_data.apply(add_team_venue_pair, axis=1)
test_data['Venue'] = test_data.apply(add_team_venue_pair, axis=1)

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# 7. Save processed datasets
train_data.to_csv("data/train_test/train_data_chronological.csv", index=False)
test_data.to_csv("data/train_test/test_data_chronological.csv", index=False)

print("\nData preparation complete. Files saved to data/train_test/train_data_chronological.csv and data/train_test/test_data_chronological.csv")

# Display feature columns created
print("\nFeature columns created (first 10):")
feature_cols = [col for col in model_data.columns if col.startswith('Home_') or col.startswith('Away_') or col.startswith('Diff_')]
print(feature_cols[:10])
print(f"Total features: {len(feature_cols)}")

# Display a sample of team-venue pairs
print("\nSample of team-venue pairs:")
for team, venue in list(team_venue_mapping.items())[:5]:
    print(f"{team}: {venue}") 
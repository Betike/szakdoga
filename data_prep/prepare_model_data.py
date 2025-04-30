import pandas as pd
import numpy as np

match_data = pd.read_csv("data/raw/premier_league_match_outcomes_2017-2018_to_2024-2025.csv")
team_stats = pd.read_csv("data/processed/team_performance_dataset.csv")

print(f"Match data shape: {match_data.shape}")
print(f"Team stats shape: {team_stats.shape}")

match_data['Date'] = pd.to_datetime(match_data['Date'])

match_data = match_data.sort_values('Date')

team_venue_mapping = {}
for _, row in match_data.iterrows():
    team = row['HomeTeam']
    venue = row['Venue']
    if team not in team_venue_mapping and not pd.isna(venue):
        team_venue_mapping[team] = venue

home_stats = team_stats.rename(columns={col: f'Home_{col}' for col in team_stats.columns if col != 'Squad' and col != 'Season'})
home_stats = home_stats.rename(columns={'Squad': 'HomeTeam'})

model_data = pd.merge(
    match_data,
    home_stats,
    on=['HomeTeam', 'Season'],
    how='left'
)

away_stats = team_stats.rename(columns={col: f'Away_{col}' for col in team_stats.columns if col != 'Squad' and col != 'Season'})
away_stats = away_stats.rename(columns={'Squad': 'AwayTeam'})

model_data = pd.merge(
    model_data,
    away_stats,
    on=['AwayTeam', 'Season'],
    how='left'
)

numeric_cols = team_stats.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col not in ['Squad', 'Season']:
        home_col = f'Home_{col}'
        away_col = f'Away_{col}'
        if home_col in model_data.columns and away_col in model_data.columns:
            model_data[f'Diff_{col}'] = model_data[home_col] - model_data[away_col]

print("Result distribution:")
print(model_data['Result'].value_counts())

holdout_seasons = ['2023-2024', '2024-2025']
train_data = model_data[~model_data['Season'].isin(holdout_seasons)].copy()
test_data = model_data[model_data['Season'].isin(holdout_seasons)].copy()

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

train_data.to_csv("data/train_test/train_data_chronological.csv", index=False)
test_data.to_csv("data/train_test/test_data_chronological.csv", index=False)

print("\nFiles saved to data/train_test/train_data_chronological.csv and data/train_test/test_data_chronological.csv")
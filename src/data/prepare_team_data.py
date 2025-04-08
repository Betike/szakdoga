import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_team_form(conn, team_id, match_date, n_matches=5):
    """Get team's form from previous n matches."""
    query = """
    SELECT 
        CASE 
            WHEN m.home_team_api_id = ? THEN m.home_team_goal - m.away_team_goal
            ELSE m.away_team_goal - m.home_team_goal
        END as goal_difference,
        CASE 
            WHEN m.home_team_api_id = ? THEN m.home_team_goal
            ELSE m.away_team_goal
        END as goals_scored,
        CASE 
            WHEN m.home_team_api_id = ? THEN m.away_team_goal
            ELSE m.home_team_goal
        END as goals_conceded
    FROM Match m
    WHERE (m.home_team_api_id = ? OR m.away_team_api_id = ?)
    AND m.date < ?
    AND m.date > date(?, '-1 year')  -- Only look at last year's matches
    ORDER BY m.date DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=[team_id] * 5 + [match_date, match_date, n_matches])
    
    if len(df) == 0:
        return pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    goal_diff_mean = df['goal_difference'].mean()
    goal_diff_std = df['goal_difference'].std() if len(df) > 1 else 0
    goals_scored_mean = df['goals_scored'].mean()
    goals_scored_std = df['goals_scored'].std() if len(df) > 1 else 0
    goals_conceded_mean = df['goals_conceded'].mean()
    goals_conceded_std = df['goals_conceded'].std() if len(df) > 1 else 0
    
    form_metrics = pd.Series([
        goal_diff_mean,
        goal_diff_std,
        goals_scored_mean,
        goals_scored_std,
        goals_conceded_mean,
        goals_conceded_std,
        (df['goal_difference'] > 0).mean(),
        (df['goal_difference'] == 0).mean(),
        (df['goal_difference'] < 0).mean(),
        len(df) / n_matches
    ])
    
    return form_metrics

def load_team_data():
    """Load and prepare team data with historical performance and attributes."""
    db_path = Path(__file__).parent.parent.parent / 'database.sqlite'
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        m.match_api_id,
        m.date,
        m.home_team_api_id,
        m.away_team_api_id,
        m.home_team_goal,
        m.away_team_goal,
        ht.buildUpPlaySpeed as home_buildup_speed,
        ht.buildUpPlayPassing as home_buildup_passing,
        ht.chanceCreationPassing as home_chance_creation,
        ht.defencePressure as home_defence_pressure,
        ht.defenceAggression as home_defence_aggression,
        ht.defenceTeamWidth as home_defence_width,
        at.buildUpPlaySpeed as away_buildup_speed,
        at.buildUpPlayPassing as away_buildup_passing,
        at.chanceCreationPassing as away_chance_creation,
        at.defencePressure as away_defence_pressure,
        at.defenceAggression as away_defence_aggression,
        at.defenceTeamWidth as away_defence_width
    FROM Match m
    LEFT JOIN Team_Attributes ht ON m.home_team_api_id = ht.team_api_id
    LEFT JOIN Team_Attributes at ON m.away_team_api_id = at.team_api_id
    WHERE 
        m.home_team_goal IS NOT NULL 
        AND m.away_team_goal IS NOT NULL
        AND ht.buildUpPlaySpeed IS NOT NULL
        AND at.buildUpPlaySpeed IS NOT NULL
    ORDER BY m.date DESC
    LIMIT 1000  -- Limit to most recent 1000 matches for faster processing
    """
    
    print("Loading match data...")
    df = pd.read_sql_query(query, conn)
    
    df['outcome'] = np.where(df['home_team_goal'] > df['away_team_goal'], 2,
                           np.where(df['home_team_goal'] == df['away_team_goal'], 1, 0))
    
    print("Calculating team form...")
    home_form = []
    away_form = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing matches"):
        home_form.append(get_team_form(conn, row['home_team_api_id'], row['date']))
        away_form.append(get_team_form(conn, row['away_team_api_id'], row['date']))
    
    home_form_df = pd.DataFrame(home_form, columns=[
        'home_avg_goal_diff', 'home_goal_diff_std', 'home_avg_goals_scored',
        'home_goals_scored_std', 'home_avg_goals_conceded', 'home_goals_conceded_std',
        'home_win_rate', 'home_draw_rate', 'home_loss_rate', 'home_matches_played'
    ])
    
    away_form_df = pd.DataFrame(away_form, columns=[
        'away_avg_goal_diff', 'away_goal_diff_std', 'away_avg_goals_scored',
        'away_goals_scored_std', 'away_avg_goals_conceded', 'away_goals_conceded_std',
        'away_win_rate', 'away_draw_rate', 'away_loss_rate', 'away_matches_played'
    ])
    
    feature_columns = [
        'home_buildup_speed', 'home_buildup_passing', 'home_chance_creation',
        'home_defence_pressure', 'home_defence_aggression', 'home_defence_width',
        'away_buildup_speed', 'away_buildup_passing', 'away_chance_creation',
        'away_defence_pressure', 'away_defence_aggression', 'away_defence_width',
        'home_avg_goal_diff', 'home_goal_diff_std', 'home_avg_goals_scored',
        'home_goals_scored_std', 'home_avg_goals_conceded', 'home_goals_conceded_std',
        'home_win_rate', 'home_draw_rate', 'home_loss_rate', 'home_matches_played',
        'away_avg_goal_diff', 'away_goal_diff_std', 'away_avg_goals_scored',
        'away_goals_scored_std', 'away_avg_goals_conceded', 'away_goals_conceded_std',
        'away_win_rate', 'away_draw_rate', 'away_loss_rate', 'away_matches_played'
    ]
    
    X = pd.concat([
        df[['home_buildup_speed', 'home_buildup_passing', 'home_chance_creation',
            'home_defence_pressure', 'home_defence_aggression', 'home_defence_width',
            'away_buildup_speed', 'away_buildup_passing', 'away_chance_creation',
            'away_defence_pressure', 'away_defence_aggression', 'away_defence_width']],
        home_form_df,
        away_form_df
    ], axis=1)
    
    y = df['outcome'].values
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    conn.close()
    
    return X, y, feature_columns

def prepare_data_for_training(X, y, test_size=0.2, random_state=42):
    """Prepare features and target for training."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    X, y, feature_columns = load_team_data()
    
    print("\nDataset Summary:")
    print(f"Total matches: {len(X)}")
    print("\nOutcome distribution:")
    print(pd.Series(y).value_counts(normalize=True))
    
    print("\nFeature columns:")
    for col in feature_columns:
        print(f"- {col}")
    
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(X, y)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}") 
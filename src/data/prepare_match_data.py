import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_match_data():
    """Load and prepare match data with relevant features."""
    db_path = Path(__file__).parent.parent.parent / 'database.sqlite'
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        m.match_api_id,
        m.home_team_api_id,
        m.away_team_api_id,
        m.home_team_goal,
        m.away_team_goal,
        m.B365H,  -- Home win odds
        m.B365D,  -- Draw odds
        m.B365A,  -- Away win odds
        ht.buildUpPlaySpeed as home_buildup_speed,
        ht.buildUpPlayPassing as home_buildup_passing,
        ht.chanceCreationPassing as home_chance_creation,
        ht.defencePressure as home_defence_pressure,
        at.buildUpPlaySpeed as away_buildup_speed,
        at.buildUpPlayPassing as away_buildup_passing,
        at.chanceCreationPassing as away_chance_creation,
        at.defencePressure as away_defence_pressure
    FROM Match m
    LEFT JOIN Team_Attributes ht ON m.home_team_api_id = ht.team_api_id
    LEFT JOIN Team_Attributes at ON m.away_team_api_id = at.team_api_id
    WHERE 
        m.home_team_goal IS NOT NULL 
        AND m.away_team_goal IS NOT NULL
        AND ht.buildUpPlaySpeed IS NOT NULL
        AND at.buildUpPlaySpeed IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['outcome'] = np.where(df['home_team_goal'] > df['away_team_goal'], 2,
                           np.where(df['home_team_goal'] == df['away_team_goal'], 1, 0))
    
    feature_columns = [
        'B365H', 'B365D', 'B365A',
        'home_buildup_speed', 'home_buildup_passing', 'home_chance_creation', 'home_defence_pressure',
        'away_buildup_speed', 'away_buildup_passing', 'away_chance_creation', 'away_defence_pressure'
    ]
    
    df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())
    
    return df, feature_columns

def prepare_data_for_training(df, feature_columns, test_size=0.2, random_state=42):
    """Prepare features and target for training."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns])
    y = df['outcome'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    df, feature_columns = load_match_data()
    
    print("\nDataset Summary:")
    print(f"Total matches: {len(df)}")
    print("\nOutcome distribution:")
    print(df['outcome'].value_counts(normalize=True))
    
    print("\nFeature columns:")
    for col in feature_columns:
        print(f"- {col}")
    
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(df, feature_columns)
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}") 
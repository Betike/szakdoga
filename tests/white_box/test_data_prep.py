import unittest
import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil

class TestDataPreparation(unittest.TestCase):
    
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        sys.path.insert(0, self.project_root)
        
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, "data/raw")
        self.processed_dir = os.path.join(self.test_dir, "data/processed")
        self.train_test_dir = os.path.join(self.test_dir, "data/train_test")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.train_test_dir, exist_ok=True)
        
        self.create_test_data()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def create_test_data(self):
        match_data = pd.DataFrame({
            'Season': ['2021-2022', '2021-2022', '2021-2022', '2022-2023', '2022-2023'],
            'Date': ['2021-08-14', '2021-08-21', '2021-08-28', '2022-08-06', '2022-08-13'],
            'HomeTeam': ['Team A', 'Team B', 'Team C', 'Team A', 'Team B'],
            'AwayTeam': ['Team B', 'Team C', 'Team A', 'Team C', 'Team A'],
            'HomeGoals': [2, 1, 0, 3, 2],
            'AwayGoals': [1, 1, 2, 0, 2],
            'Result': ['H', 'D', 'A', 'H', 'D'],
            'Venue': ['Stadium A', 'Stadium B', 'Stadium C', 'Stadium A', 'Stadium B'],
            'HomeXG': [1.8, 1.2, 0.5, 2.7, 2.1],
            'AwayXG': [1.1, 1.0, 1.7, 0.4, 2.0]
        })
        
        team_stats = pd.DataFrame({
            'Squad': ['Team A', 'Team B', 'Team C', 'Team A', 'Team B', 'Team C'],
            'Season': ['2021-2022', '2021-2022', '2021-2022', '2022-2023', '2022-2023', '2022-2023'],
            'MP': [38, 38, 38, 38, 38, 38],
            'W': [20, 18, 16, 22, 19, 15],
            'D': [10, 8, 10, 8, 10, 12],
            'L': [8, 12, 12, 8, 9, 11],
            'GF': [65, 60, 55, 75, 65, 50],
            'GA': [40, 45, 50, 35, 40, 45],
            'GD': [25, 15, 5, 40, 25, 5],
            'Pts': [70, 62, 58, 74, 67, 57],
            'xG': [66.5, 58.2, 53.1, 73.8, 62.5, 48.7],
            'xGA': [41.2, 43.7, 51.2, 36.8, 42.3, 44.5],
            'Possession': [55.2, 52.1, 51.3, 58.7, 53.2, 49.8],
            'Pass_Completion': [82.5, 80.1, 78.3, 84.2, 81.5, 77.9],
            'SoT_per_90': [5.2, 4.8, 4.3, 5.7, 5.0, 4.1]
        })
        
        match_data.to_csv(os.path.join(self.raw_dir, 
                                      "premier_league_match_outcomes_2021-2022_to_2022-2023.csv"), 
                          index=False)
        team_stats.to_csv(os.path.join(self.processed_dir, 
                                     "team_performance_dataset.csv"), 
                         index=False)
    
    
    def test_feature_creation(self):
        match_data = pd.read_csv(os.path.join(self.raw_dir, 
                                            "premier_league_match_outcomes_2021-2022_to_2022-2023.csv"))
        team_stats = pd.read_csv(os.path.join(self.processed_dir, 
                                           "team_performance_dataset.csv"))
        
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
        
        self.assertGreater(len(model_data.columns), len(match_data.columns),
                         "Feature creation should add columns")
        
        self.assertIn('Home_W', model_data.columns, "Home_W column should be created")
        self.assertIn('Away_D', model_data.columns, "Away_D column should be created")
        self.assertIn('Diff_xG', model_data.columns, "Diff_xG column should be created")
        
        team_a_home_wins = team_stats[(team_stats['Squad'] == 'Team A') & 
                                    (team_stats['Season'] == '2021-2022')]['W'].values[0]
        
        model_a_home_wins = model_data[(model_data['HomeTeam'] == 'Team A') & 
                                      (model_data['Season'] == '2021-2022')]['Home_W'].values[0]
        
        self.assertEqual(team_a_home_wins, model_a_home_wins,
                       "Home team wins should match between datasets")
    
    def test_train_test_split(self):
        match_data = pd.read_csv(os.path.join(self.raw_dir, 
                                            "premier_league_match_outcomes_2021-2022_to_2022-2023.csv"))
        team_stats = pd.read_csv(os.path.join(self.processed_dir, 
                                           "team_performance_dataset.csv"))
        
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
        
        holdout_seasons = ['2022-2023']
        train_data = model_data[~model_data['Season'].isin(holdout_seasons)].copy()
        test_data = model_data[model_data['Season'].isin(holdout_seasons)].copy()
        
        self.assertEqual(len(train_data), 3, "Train data should contain 3 matches")
        self.assertEqual(len(test_data), 2, "Test data should contain 2 matches")
        
        self.assertTrue(all(season == '2021-2022' for season in train_data['Season']),
                       "Train data should only contain 2021-2022 season")
        self.assertTrue(all(season == '2022-2023' for season in test_data['Season']),
                       "Test data should only contain 2022-2023 season")
    
    
    def test_data_cleaning(self):
        match_data = pd.read_csv(os.path.join(self.raw_dir, 
                                            "premier_league_match_outcomes_2021-2022_to_2022-2023.csv"))
        match_data.loc[0, 'HomeXG'] = np.nan
        match_data.to_csv(os.path.join(self.raw_dir, 
                                     "premier_league_match_outcomes_2021-2022_to_2022-2023.csv"), 
                         index=False)
        
        team_stats = pd.read_csv(os.path.join(self.processed_dir, 
                                           "team_performance_dataset.csv"))
        team_stats.loc[0, 'xG'] = np.nan
        team_stats.to_csv(os.path.join(self.processed_dir, 
                                     "team_performance_dataset.csv"), 
                         index=False)
        
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
        
        nan_count = model_data.isna().sum().sum()
        
        self.assertGreater(nan_count, 0, "Test data should contain NaN values")
        
        feature_cols = []
        numeric_cols = team_stats.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Squad', 'Season']:
                home_col = f'Home_{col}'
                away_col = f'Away_{col}'
                if (home_col in model_data.columns and away_col in model_data.columns and
                    not model_data[home_col].isna().any() and not model_data[away_col].isna().any()):
                    model_data[f'Diff_{col}'] = model_data[home_col] - model_data[away_col]
                    feature_cols.append(f'Diff_{col}')
        
        self.assertNotIn('Diff_xG', feature_cols, 
                       "Diff_xG should not be created because source data has NaN")

if __name__ == '__main__':
    unittest.main() 
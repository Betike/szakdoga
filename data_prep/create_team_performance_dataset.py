import pandas as pd
import os

# Directory where your CSV files are located
raw_data_dir = "data/raw"
data_dir = "data/processed"

def load_league_table():
    """
    Load the league table data with home and away statistics
    """
    print("Loading league table data...")
    league_file = os.path.join(raw_data_dir, "all_seasons_league_table.csv")
    
    if os.path.exists(league_file):
        league_data = pd.read_csv(league_file)
        print(f"Loaded league table: {league_data.shape}")
        return league_data
    else:
        print(f"Error: League table file not found: {league_file}")
        return None

def load_team_stats():
    """
    Load additional team statistics from the combined tables
    """
    print("Loading additional team statistics...")
    
    # Dictionary to store all tables
    stats = {}
    
    # Load each type of statistic
    file_types = [
        "all_seasons_standard_stats",
        "all_seasons_shooting",
        "all_seasons_passing",
        "all_seasons_defensive",
        "all_seasons_possession"
    ]
    
    for file_type in file_types:
        file_path = os.path.join(raw_data_dir, f"{file_type}.csv")
        if os.path.exists(file_path):
            stats[file_type] = pd.read_csv(file_path)
            print(f"Loaded {file_type}: {stats[file_type].shape}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    return stats

def create_team_performance_dataset(league_data, team_stats):
    """
    Create team performance dataset by combining league table data with other stats
    """
    print("\nCreating team performance dataset...")
    
    # Start with league data as the base
    performance_data = league_data.copy()
    
    # Add derived features from league data
    performance_data['HomeWinRate'] = performance_data['W'] / performance_data['MP']
    performance_data['AwayWinRate'] = performance_data['A_W'] / performance_data['A_MP']
    performance_data['HomeGoalsPerMatch'] = performance_data['GF'] / performance_data['MP']
    performance_data['AwayGoalsPerMatch'] = performance_data['A_GF'] / performance_data['A_MP']
    performance_data['HomeGoalsConcededPerMatch'] = performance_data['GA'] / performance_data['MP']
    performance_data['AwayGoalsConcededPerMatch'] = performance_data['A_GA'] / performance_data['A_MP']
    performance_data['HomePointsPerMatch'] = performance_data['Pts'] / performance_data['MP']
    performance_data['AwayPointsPerMatch'] = performance_data['A_Pts'] / performance_data['A_MP']
    
    # Calculate overall statistics
    performance_data['TotalMP'] = performance_data['MP'] + performance_data['A_MP']
    performance_data['TotalW'] = performance_data['W'] + performance_data['A_W']
    performance_data['TotalD'] = performance_data['D'] + performance_data['A_D']
    performance_data['TotalL'] = performance_data['L'] + performance_data['A_L']
    performance_data['TotalGF'] = performance_data['GF'] + performance_data['A_GF']
    performance_data['TotalGA'] = performance_data['GA'] + performance_data['A_GA']
    performance_data['TotalGD'] = performance_data['GD'] + performance_data['A_GD']
    performance_data['TotalPts'] = performance_data['Pts'] + performance_data['A_Pts']
    
    performance_data['OverallWinRate'] = performance_data['TotalW'] / performance_data['TotalMP']
    performance_data['OverallGoalsPerMatch'] = performance_data['TotalGF'] / performance_data['TotalMP']
    performance_data['OverallGoalsConcededPerMatch'] = performance_data['TotalGA'] / performance_data['TotalMP']
    performance_data['OverallPointsPerMatch'] = performance_data['TotalPts'] / performance_data['TotalMP']
    
    # Calculate home vs away performance differences
    performance_data['HomeAwayWinRateDiff'] = performance_data['HomeWinRate'] - performance_data['AwayWinRate']
    performance_data['HomeAwayGoalsScoredDiff'] = performance_data['HomeGoalsPerMatch'] - performance_data['AwayGoalsPerMatch']
    performance_data['HomeAwayGoalsConcededDiff'] = performance_data['HomeGoalsConcededPerMatch'] - performance_data['AwayGoalsConcededPerMatch']
    performance_data['HomeAwayPointsDiff'] = performance_data['HomePointsPerMatch'] - performance_data['AwayPointsPerMatch']
    
    # Merge with other statistics if available
    if team_stats:
        # Try to merge with standard stats
        standard_stats = team_stats.get("all_seasons_standard_stats")
        standard_stats_cols = ['Squad', 'Season', 'Poss', 'PrgC', 'PrgP']
        if standard_stats is not None:
            join_cols = ['Squad', 'Season'] if 'Season' in standard_stats.columns else ['Squad']
            performance_data = pd.merge(
                performance_data, 
                standard_stats[standard_stats_cols],
                on=join_cols,
                how='left',
                suffixes=('', '_std')
            )
        
        # Try to merge with shooting stats
        shooting_stats = team_stats.get("all_seasons_shooting")
        shooting_stats_cols = ['Squad', 'Season', 'Sh', 'SoT', 'SoT%']
        if shooting_stats is not None:
            join_cols = ['Squad', 'Season'] if 'Season' in shooting_stats.columns else ['Squad']
            performance_data = pd.merge(
                performance_data, 
                shooting_stats[shooting_stats_cols],
                on=join_cols,
                how='left',
                suffixes=('', '_shooting')
            )
        
        # Try to merge with possession stats
        possession_stats = team_stats.get("all_seasons_possession")
        possession_stats_cols = ['Squad', 'Season', 'Succ', 'Touches', 'Att 3rd', 'Att Pen', 'Live']
        if possession_stats is not None:
            join_cols = ['Squad', 'Season'] if 'Season' in possession_stats.columns else ['Squad']
            performance_data = pd.merge(
                performance_data, 
                possession_stats[possession_stats_cols],
                on=join_cols,
                how='left',
                suffixes=('', '_possession')
            )
    
    performance_data_df=pd.DataFrame(performance_data)
    
    return performance_data_df

def main():
    # Load league table data
    league_data = load_league_table()
    
    if league_data is None:
        print("Error: Could not load league table data. Exiting.")
        return
    
    # Load additional team statistics
    team_stats = load_team_stats()
    
    # Create team performance dataset
    team_performance = create_team_performance_dataset(league_data, team_stats)
    
    # Save team performance dataset
    team_performance_file = os.path.join(data_dir, "team_performance_dataset.csv")
    team_performance.to_csv(team_performance_file, index=False)
    print(f"Saved team performance dataset to {team_performance_file}")
    print(f"Shape: {team_performance.shape}")

if __name__ == "__main__":
    main() 
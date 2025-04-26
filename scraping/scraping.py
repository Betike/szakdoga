# %%
# Import required libraries
import pandas as pd
import io
import os
import time
import traceback
import requests


csv_dir = "data/raw"
print(csv_dir)


# %% 
# Function to wait a bit between requests to avoid rate limiting
def wait_between_requests(seconds=2):
    print(f"Waiting {seconds} seconds before next request...")
    time.sleep(seconds)

# %%
def scrape_premier_league_detailed_stats(season="2023-2024", save_individual_tables=True):
    """
    Scrape detailed Premier League stats tables from FBRef
    
    Args:
        season: The season to scrape (e.g., '2023-2024')
        save_individual_tables: Whether to save each table as a separate CSV
    
    Returns:
        Dictionary containing all the scraped tables
    """
    print(f"\n=== Scraping detailed Premier League stats for {season} ===")
    
    # Construct the URL for the season's stats page
    if season == "2024-2025":
        url = f"https://fbref.com/en/comps/9/Premier-League-Stats"
    else:
        url = f"https://fbref.com/en/comps/9/{season}/{season}-Premier-League-Stats"
    
    # Get the page
    payload = { 'api_key': 'f3188e6e1dfe4d89a2c6602ef682d74e', 'url': url }
    response = requests.get('https://api.scraperapi.com/', params=payload)
    
    if response.status_code != 200:
        print(f"Failed to get data for {season}: Status code {response.status_code}")
        return None
    
    # Dictionary to store all tables
    tables_dict = {}

    try:
        all_tables = pd.read_html(io.StringIO(response.text), header=1)
        print(f"Found {len(all_tables)} tables on the page")

        if len(all_tables) >= 19:  # Make sure we have enough tables
            tables_dict["league_table"] = all_tables[1]
            tables_dict["standard_stats"] = all_tables[2]
            tables_dict["goalkeeping"] = all_tables[4]
            tables_dict["shooting"] = all_tables[8]
            tables_dict["passing"] = all_tables[10]
            tables_dict["goal_creation"] = all_tables[14]
            tables_dict["defensive"] = all_tables[16]
            tables_dict["possession"] = all_tables[18]

            # Clean up column names for league table
            lt = tables_dict["league_table"]
            league_renames = {
                col: f"A_{col.split('.')[0]}"
                for col in lt.columns
                if col.endswith(".1")
            }
            lt.rename(columns=league_renames, inplace=True)

            # Clean up column names for goal creation stats
            gc = tables_dict["goal_creation"]
            gc_renames = {
                col: col.replace(".1","_G")
                for col in gc.columns
                if col.endswith(".1")
            }
            gc.rename(columns=gc_renames, inplace=True)

            # Clean up column names for goalkeeping stats
            tables_dict["goalkeeping"].rename(columns={'Save%.1': 'PKSave%'}, inplace = True)

            # Drop duplicate columns for passing stats
            tables_dict["passing"].drop(columns=["Cmp.1","Att.1","Cmp%.1","Cmp.2","Att.2","Cmp%.2","Cmp.3","Att.3","Cmp%.3"], inplace = True, errors='ignore')

            # Drop duplicate columns for standard stats
            tables_dict["standard_stats"].drop(columns=["Gls.1","Ast.1","G+A.1","G-PK.1","G+A-PK","xG.1","xAG.1","xG+xAG","npxG.1","npxG+xAG.1"], inplace = True, errors='ignore')

            # Add season column to each table and save
            for table_name, table in tables_dict.items():
                table.insert(0, 'Season', season)
                if save_individual_tables:
                    table_path = os.path.join(csv_dir, f"{season}_{table_name}.csv")
                    table.to_csv(table_path, index=False)
                    print(f"Saved {table_name} to {table_path}")   
        else:
            print(f"Warning: Found fewer tables ({len(all_tables)}) than expected (19+) for season {season}")
            return None
                  
    except Exception as e:
        print(f"Error processing data for {season}: {str(e)}")
        return None
                  
    return tables_dict

# %%
def scrape_multiple_seasons(seasons, save_combined_tables=True):
    """
    Scrape multiple seasons and combine the results
    
    Args:
        seasons: List of seasons to scrape (e.g., ['2022-2023', '2023-2024'])
        save_combined_tables: Whether to save combined tables across seasons
    
    Returns:
        Dictionary of combined tables
    """
    # Dictionary to store tables for each category across seasons
    combined_tables = {
        "league_table": [],
        "standard_stats": [],
        "goalkeeping": [],
        "shooting": [],
        "passing": [],
        "goal_creation": [],
        "defensive": [],
        "possession": []
    }
    
    # Process each season
    for season in seasons:
        print(f"\nProcessing season: {season}")
        
        # Scrape data for this season
        season_tables = scrape_premier_league_detailed_stats(season=season, save_individual_tables=False)
        
        if not season_tables:
            print(f"Skipping season {season} due to errors")
            continue
            
        # Add each table to the combined collection
        for table_name, table in season_tables.items():
            if table_name in combined_tables:
                combined_tables[table_name].append(table)
        
        # Wait between seasons to avoid overloading the API
        if season != seasons[-1]:  # Don't wait after the last season
            wait_between_requests(3)
    
    # Combine tables across seasons
    result = {}
    if save_combined_tables:
        for table_name, tables_list in combined_tables.items():
            if tables_list:
                # Combine all seasons into one DataFrame
                combined_df = pd.concat(tables_list, ignore_index=True)
                result[table_name] = combined_df
                
                # Save the combined table
                combined_path = os.path.join(csv_dir, f"all_seasons_{table_name}.csv")
                combined_df.to_csv(combined_path, index=False)
                print(f"Saved combined {table_name} table with {len(combined_df)} rows to {combined_path}")
    
    return result

# %%
def scrape_premier_league_match_outcomes(seasons=None):
    """
    Scrape actual match outcomes from the Premier League Scores & Fixtures table
    
    Args:
        seasons (list, optional): List of seasons to scrape (format: '2023-2024').
            If None, uses a default set of recent seasons.
    
    Returns:
        pd.DataFrame: DataFrame containing match outcome data for all requested seasons
    """
    if seasons is None:
        # Default to recent seasons if none provided
        seasons = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', 
                  '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    
    print(f"Scraping match outcomes for {len(seasons)} seasons...")
    all_matches = []
    
    for season in seasons:
        print(f"Scraping match outcomes for season {season}...")
        
        # Construct the URL for the season's scores & fixtures
        if season == "2024-2025":
            url = f"https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
        else:
            url = f"https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"
        
        try:
            # Use the ScraperAPI to get the page content
            payload = {'api_key': 'f3188e6e1dfe4d89a2c6602ef682d74e', 'url': url}
            response = requests.get('https://api.scraperapi.com/', params=payload)
            
            if response.status_code != 200:
                print(f"Failed to retrieve data for season {season}: Status code {response.status_code}")
                continue
            
            # Parse the HTML tables using pandas
            try:
                tables = pd.read_html(io.StringIO(response.text))
                print(f"Found {len(tables)} tables on the page")
                
                # Find the scores & fixtures table (usually the first large table)
                fixtures_table = None
                for table in tables:
                    # Check if this table has the expected columns for match data
                    if (isinstance(table, pd.DataFrame) and 
                        len(table) > 30 and  # Usually has many rows
                        'Home' in table.columns and 
                        'Away' in table.columns and
                        'Score' in str(table.columns)):  # Score might be part of a multi-index
                        fixtures_table = table
                        break
                
                if fixtures_table is None:
                    print(f"Could not identify the scores & fixtures table for {season}")
                    continue
                
                # Clean up the table - handle potential multi-index columns
                if isinstance(fixtures_table.columns, pd.MultiIndex):
                    # Flatten multi-index by joining with underscore
                    fixtures_table.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in fixtures_table.columns]
                
                # Find the score column - it might have different names
                score_col = next((col for col in fixtures_table.columns if 'Score' in col), None)
                if not score_col:
                    print(f"Could not find Score column in the table for {season}")
                    continue

                xg_cols = [c for c in fixtures_table.columns if c.startswith('xG')]
                if len(xg_cols) >= 2:
                    fixtures_table = fixtures_table.rename(
                        columns={xg_cols[0]: 'HomeXG', xg_cols[1]: 'AwayXG'}
                    )
                
                # Process each match
                for _, row in fixtures_table.iterrows():
                    # Skip rows without a score (matches not yet played)
                    score_text = str(row[score_col]).strip()
                    if pd.isna(row[score_col]) or score_text == '' or '–' not in score_text:
                        continue
                    
                    # Extract match data
                    try:
                        # Extract basic match information
                        date = row.get('Date', '')
                        home_team = row['Home']
                        away_team = row['Away']
                        
                        # Process score
                        home_goals, away_goals = map(int, score_text.split('–'))
                        
                        # Determine match result
                        if home_goals > away_goals:
                            result = 'H'  # Home win
                        elif home_goals < away_goals:
                            result = 'A'  # Away win
                        else:
                            result = 'D'  # Draw
                            
                        # Create match data dictionary
                        match_data = {
                            'Season': season,
                            'Date': date,
                            'HomeTeam': home_team,
                            'AwayTeam': away_team,
                            'HomeGoals': home_goals,
                            'AwayGoals': away_goals,
                            'Result': result,
                        }
                        
                        # Add time if available
                        if 'Time' in row:
                            match_data['Time'] = row['Time']
                            
                        # Add venue if available
                        if 'Venue' in row:
                            match_data['Venue'] = row['Venue']
                            
                        # Add xG data if available
                        match_data['HomeXG'] = row.get('HomeXG')  # returns None if the cell is empty / NaN
                        match_data['AwayXG'] = row.get('AwayXG')
                        
                        all_matches.append(match_data)
                    except Exception as e:
                        print(f"Error processing match in {season}: {e}")
                        continue
                
                print(f"Processed {len(fixtures_table)} rows, extracted {sum(1 for match in all_matches if match['Season'] == season)} matches for season {season}")
                
            except Exception as e:
                print(f"Error parsing HTML tables for {season}: {e}")
                traceback.print_exc()
                continue
            
            # Respect the website's rate limits
            wait_between_requests(3)
            
        except Exception as e:
            print(f"Error scraping season {season}: {e}")
            traceback.print_exc()

    print(f"Scraping match outcomes for season {season}...")
    
    # Convert to DataFrame
    matches_df = pd.DataFrame(all_matches)
    
    # Save to CSV
    scores_path = os.path.join(csv_dir, f"premier_league_match_outcomes_{seasons[0]}_to_2024-2025.csv")
    matches_df.to_csv(scores_path, index=False)
    print(f"Saved match outcomes data to {scores_path}")
    print(f"Total matches scraped: {len(matches_df)}")
    
    return matches_df

if __name__ == "__main__":
    # Example usage of the new functions
    seasons_to_scrape = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', 
                         '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    # Scrape match outcomes
    match_outcomes = scrape_premier_league_match_outcomes(seasons=seasons_to_scrape)
    # Scrape detailed stats
    detailed_stats = scrape_multiple_seasons(seasons=seasons_to_scrape)
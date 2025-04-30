import pandas as pd
import io
import os
import time
import traceback
import requests


csv_dir = "data/raw"

def wait_between_requests(seconds=2):
    print(f"Waiting {seconds} seconds before next request...")
    time.sleep(seconds)

def scrape_detailed_stats(season="2023-2024", save_individual_tables=True):
    print(f"\nScraping stats for {season}")

    if season == "2024-2025":
        url = f"https://fbref.com/en/comps/9/Premier-League-Stats"
    else:
        url = f"https://fbref.com/en/comps/9/{season}/{season}-Premier-League-Stats"

    # You have to get your own API key from ScraperAPI replace the *api_key* with your own
    payload = { 'api_key': '*api_key*', 'url': url }
    response = requests.get('https://api.scraperapi.com/', params=payload)
    
    if response.status_code != 200:
        print(f"Failed to get data for {season}: Status code {response.status_code}")
        return None

    tables_dict = {}

    try:
        all_tables = pd.read_html(io.StringIO(response.text), header=1)

        if len(all_tables) >= 19:
            tables_dict["league_table"] = all_tables[1]
            tables_dict["standard_stats"] = all_tables[2]
            tables_dict["goalkeeping"] = all_tables[4]
            tables_dict["shooting"] = all_tables[8]
            tables_dict["passing"] = all_tables[10]
            tables_dict["goal_creation"] = all_tables[14]
            tables_dict["defensive"] = all_tables[16]
            tables_dict["possession"] = all_tables[18]

            lt = tables_dict["league_table"]
            league_renames = {
                col: f"A_{col.split('.')[0]}"
                for col in lt.columns
                if col.endswith(".1")
            }
            lt.rename(columns=league_renames, inplace=True)

            gc = tables_dict["goal_creation"]
            gc_renames = {
                col: col.replace(".1","_G")
                for col in gc.columns
                if col.endswith(".1")
            }
            gc.rename(columns=gc_renames, inplace=True)

            tables_dict["goalkeeping"].rename(columns={'Save%.1': 'PKSave%'}, inplace = True)

            tables_dict["passing"].drop(columns=["Cmp.1","Att.1","Cmp%.1","Cmp.2","Att.2","Cmp%.2","Cmp.3","Att.3","Cmp%.3"], inplace = True, errors='ignore')

            tables_dict["standard_stats"].drop(columns=["Gls.1","Ast.1","G+A.1","G-PK.1","G+A-PK","xG.1","xAG.1","xG+xAG","npxG.1","npxG+xAG.1"], inplace = True, errors='ignore')

            for table_name, table in tables_dict.items():
                table.insert(0, 'Season', season)
                if save_individual_tables:
                    table_path = os.path.join(csv_dir, f"{season}_{table_name}.csv")
                    table.to_csv(table_path, index=False)
                    print(f"Saved {table_name} to {table_path}")   
        else:
            print(f"Warning: Found fewer tables ({len(all_tables)}) than expected for season {season}")
            return None
                  
    except Exception as e:
        print(f"Error processing data for {season}: {str(e)}")
        return None
                  
    return tables_dict

def scrape_multiple_seasons(seasons, save_combined_tables=True):

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
    
    for season in seasons:
        print(f"\nProcessing season: {season}")
        
        season_tables = scrape_detailed_stats(season=season, save_individual_tables=False)
        
        if not season_tables:
            print(f"Skipping season {season} due to errors")
            continue
            
        for table_name, table in season_tables.items():
            if table_name in combined_tables:
                combined_tables[table_name].append(table)
        
        if season != seasons[-1]:
            wait_between_requests(3)

    result = {}
    if save_combined_tables:
        for table_name, tables_list in combined_tables.items():
            if tables_list:
                combined_df = pd.concat(tables_list, ignore_index=True)
                result[table_name] = combined_df
                combined_path = os.path.join(csv_dir, f"all_seasons_{table_name}.csv")
                combined_df.to_csv(combined_path, index=False)
                print(f"Saved combined {table_name} table with {len(combined_df)} rows to {combined_path}")
    
    return result

def scrape_match_outcomes(seasons=None):
    if seasons is None:
        seasons = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', 
                  '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    
    print(f"Scraping matches of {len(seasons)} seasons")
    all_matches = []
    
    for season in seasons:
        print(f"Scraping match outcomes for season {season}...")

        if season == "2024-2025":
            url = f"https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
        else:
            url = f"https://fbref.com/en/comps/9/{season}/schedule/{season}-Premier-League-Scores-and-Fixtures"
        
        try:
            # You have to get your own API key from ScraperAPI replace the *api_key* with your own
            payload = {'api_key': '*api_key*', 'url': url}
            response = requests.get('https://api.scraperapi.com/', params=payload)
            
            if response.status_code != 200:
                print(f"Failed to retrieve data for season {season}: Status code {response.status_code}")
                continue
            
            try:
                tables = pd.read_html(io.StringIO(response.text))
                fixtures_table = None
                for table in tables:
                    if (isinstance(table, pd.DataFrame) and 
                        len(table) > 30 and
                        'Home' in table.columns and 
                        'Away' in table.columns and
                        'Score' in str(table.columns)):
                        fixtures_table = table
                        break
                
                if fixtures_table is None:
                    print(f"Could not get the fixtures table for {season}")
                    continue
                
                if isinstance(fixtures_table.columns, pd.MultiIndex):
                    fixtures_table.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in fixtures_table.columns]
                
                score_col = next((col for col in fixtures_table.columns if 'Score' in col), None)
                if not score_col:
                    print(f"Could not find Score column in the table for {season}")
                    continue

                xg_cols = [c for c in fixtures_table.columns if c.startswith('xG')]
                if len(xg_cols) >= 2:
                    fixtures_table = fixtures_table.rename(
                        columns={xg_cols[0]: 'HomeXG', xg_cols[1]: 'AwayXG'}
                    )
                
                for _, row in fixtures_table.iterrows():
                    score_text = str(row[score_col]).strip()
                    if pd.isna(row[score_col]) or score_text == '' or '–' not in score_text:
                        continue
                    
                    try:
                        date = row.get('Date', '')
                        home_team = row['Home']
                        away_team = row['Away']
                        
                        home_goals, away_goals = map(int, score_text.split('–'))
                        
                        if home_goals > away_goals:
                            result = 'H'
                        elif home_goals < away_goals:
                            result = 'A'
                        else:
                            result = 'D'
                            
                        match_data = {
                            'Season': season,
                            'Date': date,
                            'HomeTeam': home_team,
                            'AwayTeam': away_team,
                            'HomeGoals': home_goals,
                            'AwayGoals': away_goals,
                            'Result': result,
                        }
                        
                        if 'Time' in row:
                            match_data['Time'] = row['Time']
                            
                        if 'Venue' in row:
                            match_data['Venue'] = row['Venue']
                            
                        match_data['HomeXG'] = row.get('HomeXG')
                        match_data['AwayXG'] = row.get('AwayXG')
                        
                        all_matches.append(match_data)
                    except Exception as e:
                        print(f"Error in processing {season}: {e}")
                        continue
                
                print(f"Processed {len(fixtures_table)} rows and saved {sum(1 for match in all_matches if match['Season'] == season)} matches for season {season}")
                
            except Exception as e:
                print(f"Error parsing HTML tables for {season}: {e}")
                traceback.print_exc()
                continue
            wait_between_requests(3)
            
        except Exception as e:
            print(f"Error scraping season {season}: {e}")
            traceback.print_exc()
    
    matches_df = pd.DataFrame(all_matches)
    
    scores_path = os.path.join(csv_dir, f"premier_league_match_outcomes_{seasons[0]}_to_2024-2025.csv")
    matches_df.to_csv(scores_path, index=False)
    print(f"Saved matches to {scores_path}")
    print(f"Total matches: {len(matches_df)}")
    
    return matches_df

if __name__ == "__main__":
    seasons_to_scrape = ['2017-2018', '2018-2019', '2019-2020', '2020-2021', 
                         '2021-2022', '2022-2023', '2023-2024', '2024-2025']
    match_outcomes = scrape_match_outcomes(seasons=seasons_to_scrape)
    detailed_stats = scrape_multiple_seasons(seasons=seasons_to_scrape)
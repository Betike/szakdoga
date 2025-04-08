import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def connect_to_database():
    """Connect to the SQLite database."""
    db_path = Path(__file__).parent.parent.parent / 'database.sqlite'
    return sqlite3.connect(db_path)

def get_table_info(conn):
    """Get information about all tables in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    table_info = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        
        table_info[table_name] = {
            'columns': [col[1] for col in columns],
            'types': [col[2] for col in columns],
            'row_count': row_count
        }
    
    return table_info

def print_database_summary(table_info):
    """Print a summary of the database structure."""
    print("\n=== Database Summary ===\n")
    for table_name, info in table_info.items():
        print(f"\nTable: {table_name}")
        print(f"Number of rows: {info['row_count']}")
        print("\nColumns:")
        for col, type_ in zip(info['columns'], info['types']):
            print(f"  - {col} ({type_})")
        print("-" * 50)

def main():
    """Main function to explore the database."""
    try:
        conn = connect_to_database()
        print("Successfully connected to the database.")
        
        table_info = get_table_info(conn)
        print_database_summary(table_info)
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main() 
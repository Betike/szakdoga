import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_team_data import load_team_data, prepare_data_for_training
from models.team_based_predictor import TeamBasedPredictor

def calculate_feature_importance(model, X_test, y_test, feature_names, n_repeats=5):
    """
    Calculate feature importance using feature permutation.
    This technique randomly shuffles each feature and measures the decrease in model performance.
    Features that cause larger drops when shuffled are more important.
    """
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Get baseline accuracy
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        baseline_accuracy = np.mean(predicted.numpy() == y_test)
    
    # Calculate importance for each feature
    importances = np.zeros((n_repeats, len(feature_names)))
    
    for i in range(n_repeats):
        for j in range(len(feature_names)):
            # Create a copy of the test data
            X_test_permuted = X_test.copy()
            
            # Shuffle the feature
            np.random.shuffle(X_test_permuted[:, j])
            
            # Evaluate on permuted data
            X_test_permuted_tensor = torch.FloatTensor(X_test_permuted)
            with torch.no_grad():
                outputs = model(X_test_permuted_tensor)
                _, predicted = torch.max(outputs.data, 1)
                permuted_accuracy = np.mean(predicted.numpy() == y_test)
            
            # Calculate importance as the drop in accuracy
            importances[i, j] = baseline_accuracy - permuted_accuracy
    
    # Average importance over repeats
    mean_importances = np.mean(importances, axis=0)
    
    # Scale to percentages
    total_importance = np.sum(mean_importances)
    if total_importance <= 0:  # Avoid division by zero
        percentage_importances = np.zeros(len(feature_names))
    else:
        percentage_importances = (mean_importances / total_importance) * 100
    
    # Create a DataFrame with feature names and importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_importances,
        'Percentage': percentage_importances
    })
    
    return importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

def group_features_by_category(importance_df):
    """Group features by category to understand which types of attributes are most important"""
    
    # Create category map
    category_map = {}
    
    # Home team tactical attributes
    for i, feat in enumerate(['home_buildup_speed', 'home_buildup_passing', 'home_chance_creation',
                            'home_defence_pressure', 'home_defence_aggression', 'home_defence_width']):
        category_map[feat] = 'Home Team Attributes'
    
    # Away team tactical attributes
    for feat in ['away_buildup_speed', 'away_buildup_passing', 'away_chance_creation',
                'away_defence_pressure', 'away_defence_aggression', 'away_defence_width']:
        category_map[feat] = 'Away Team Attributes'
    
    # Home team form
    for i in range(10):
        if i < 6:  # The first 6 are goal statistics
            category_map[f'home_form_{i}'] = 'Home Team Goal Stats'
        else:  # The last 4 are win/draw/loss rates
            category_map[f'home_form_{i}'] = 'Home Team Match Results'
    
    # Away team form
    for i in range(10):
        if i < 6:  # The first 6 are goal statistics
            category_map[f'away_form_{i}'] = 'Away Team Goal Stats'
        else:  # The last 4 are win/draw/loss rates
            category_map[f'away_form_{i}'] = 'Away Team Match Results'
    
    # Add category to dataframe
    importance_df['Category'] = importance_df['Feature'].map(category_map)
    
    # Group by category and sum importance
    category_importance = importance_df.groupby('Category')['Percentage'].sum().reset_index()
    
    return category_importance.sort_values('Percentage', ascending=False)

def plot_feature_importance(importance_df, output_path='visualizations/feature_importance.png'):
    """Plot individual feature importance"""
    plt.figure(figsize=(12, 10))
    
    # Plot top 20 features
    top_features = importance_df.head(20)
    
    # Create a color map for different feature categories
    category_colors = {
        'Home Team Attributes': '#ff9999',
        'Away Team Attributes': '#99ff99',
        'Home Team Goal Stats': '#9999ff',
        'Away Team Goal Stats': '#ffff99',
        'Home Team Match Results': '#ff99ff',
        'Away Team Match Results': '#99ffff'
    }
    
    colors = [category_colors.get(cat, '#cccccc') for cat in top_features['Category']]
    
    # Create the bar plot
    plt.figure(figsize=(14, 8))
    bars = plt.barh(top_features['Feature'], top_features['Percentage'], color=colors)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat) 
                      for cat, color in category_colors.items()]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.xlabel('Importance (%)')
    plt.title('Top 20 Features by Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest value at the top
    
    # Add percentage labels to bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.5
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                va='center')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def plot_category_importance(category_importance, output_path='visualizations/category_importance.png'):
    """Plot feature category importance"""
    plt.figure(figsize=(10, 6))
    
    # Create a bar plot
    bars = plt.bar(category_importance['Category'], category_importance['Percentage'])
    
    plt.xlabel('Feature Category')
    plt.ylabel('Importance (%)')
    plt.title('Feature Category Importance')
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def main():
    # Load the trained model
    model_path = 'models/team_based_predictor.pth'
    model = TeamBasedPredictor.load_model(model_path)
    model.eval()
    
    # Load and prepare data
    print("Loading data...")
    X, y, feature_columns = load_team_data()
    X_train, X_test, y_train, y_test, _ = prepare_data_for_training(X, y)
    
    # Create feature names (they were not explicitly saved in the original code)
    feature_names = [
        # Home team tactical attributes
        'home_buildup_speed', 'home_buildup_passing', 'home_chance_creation',
        'home_defence_pressure', 'home_defence_aggression', 'home_defence_width',
        
        # Away team tactical attributes
        'away_buildup_speed', 'away_buildup_passing', 'away_chance_creation',
        'away_defence_pressure', 'away_defence_aggression', 'away_defence_width'
    ]
    
    # Add form feature names
    for i in range(10):
        feature_names.append(f'home_form_{i}')
    for i in range(10):
        feature_names.append(f'away_form_{i}')
    
    # Calculate feature importance
    print("Calculating feature importance (this may take a few minutes)...")
    importance_df = calculate_feature_importance(model, X_test, y_test, feature_names, n_repeats=3)
    
    # Group features by category
    category_importance = group_features_by_category(importance_df)
    
    # Plot results
    print("Plotting results...")
    feature_plot_path = plot_feature_importance(importance_df)
    category_plot_path = plot_category_importance(category_importance)
    
    # Print top features
    print("\nTop 10 most important features:")
    print(importance_df.head(10).to_string(index=False))
    
    print("\nFeature importance by category:")
    print(category_importance.to_string(index=False))
    
    print(f"\nFeature importance plot saved to: {feature_plot_path}")
    print(f"Category importance plot saved to: {category_plot_path}")

if __name__ == "__main__":
    main() 
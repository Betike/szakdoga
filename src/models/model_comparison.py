import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_team_data import load_team_data, prepare_data_for_training
from models.team_based_predictor import TeamBasedPredictor
from models.attention_team_predictor import AttentionTeamPredictor

def load_models():
    """Load both models from saved files."""
    model_dir = Path(__file__).parent.parent.parent / 'models'
    
    # Load team-based model
    team_model_path = model_dir / 'team_based_predictor.pth'
    if team_model_path.exists():
        team_model = TeamBasedPredictor.load_model(team_model_path)
    else:
        print(f"Team-based model not found at {team_model_path}")
        team_model = None
    
    # Load attention-based model
    attention_model_path = model_dir / 'attention_team_predictor.pth'
    if attention_model_path.exists():
        attention_model = AttentionTeamPredictor.load_model(attention_model_path)
    else:
        print(f"Attention-based model not found at {attention_model_path}")
        attention_model = None
    
    return team_model, attention_model

def evaluate_model_performance(model, X_test, y_test, model_name):
    """Evaluate model performance and return metrics."""
    start_time = time.time()
    
    # Convert data to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
    
    # Calculate metrics
    y_pred = predictions.numpy()
    inference_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    # Calculate per-class metrics
    class_names = ['Away Win', 'Draw', 'Home Win']
    class_metrics = []
    
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        class_pred = y_pred[class_mask]
        class_true = y_test[class_mask]
        
        if len(class_true) > 0:
            class_acc = accuracy_score(class_true, class_pred)
            class_metrics.append({
                'class': class_name,
                'accuracy': class_acc,
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': np.sum(class_mask)
            })
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics,
        'confusion_matrix': cm,
        'inference_time': inference_time
    }

def compare_models(team_model, attention_model, X_test, y_test):
    """Compare the performance of both models."""
    results = {}
    
    if team_model:
        print("\nEvaluating Team-Based Predictor...")
        results['team'] = evaluate_model_performance(team_model, X_test, y_test, "Team-Based Predictor")
    
    if attention_model:
        print("\nEvaluating Attention-Based Predictor...")
        results['attention'] = evaluate_model_performance(attention_model, X_test, y_test, "Attention-Based Predictor")
    
    return results

def plot_comparison_metrics(results):
    """Plot comparison metrics between models."""
    vis_dir = Path(__file__).parent.parent.parent / 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Only proceed if we have results for both models
    if 'team' not in results or 'attention' not in results:
        print("Cannot plot comparison - missing model results")
        return
    
    # Extract model names
    model_names = [results['team']['model_name'], results['attention']['model_name']]
    
    # Create dataframe for overall metrics
    metrics = ['accuracy']
    overall_data = []
    
    for model_type, res in results.items():
        overall_data.append({
            'Model': res['model_name'],
            'Accuracy': res['accuracy'],
            'Inference Time (s)': res['inference_time']
        })
    
    overall_df = pd.DataFrame(overall_data)
    
    # Create dataframe for class metrics
    class_data = []
    for model_type, res in results.items():
        for class_metric in res['class_metrics']:
            class_data.append({
                'Model': res['model_name'],
                'Class': class_metric['class'],
                'Accuracy': class_metric['accuracy'],
                'Precision': class_metric['precision'],
                'Recall': class_metric['recall'],
                'F1 Score': class_metric['f1']
            })
    
    class_df = pd.DataFrame(class_data)
    
    # Plot overall accuracy comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='Accuracy', data=overall_df)
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    
    plt.subplot(2, 2, 2)
    sns.barplot(x='Model', y='Inference Time (s)', data=overall_df)
    plt.title('Inference Time Comparison')
    
    # Plot per-class F1 scores
    plt.subplot(2, 2, 3)
    sns.barplot(x='Class', y='F1 Score', hue='Model', data=class_df)
    plt.title('F1 Score by Class')
    plt.ylim(0, 1)
    
    # Plot per-class accuracy
    plt.subplot(2, 2, 4)
    sns.barplot(x='Class', y='Accuracy', hue='Model', data=class_df)
    plt.title('Accuracy by Class')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'model_comparison.png')
    plt.close()
    
    # Plot confusion matrices
    plt.figure(figsize=(16, 6))
    class_names = ['Away Win', 'Draw', 'Home Win']
    
    plt.subplot(1, 2, 1)
    plot_cm(results['team']['confusion_matrix'], class_names, 'Team-Based Predictor')
    
    plt.subplot(1, 2, 2)
    plot_cm(results['attention']['confusion_matrix'], class_names, 'Attention-Based Predictor')
    
    plt.tight_layout()
    plt.savefig(vis_dir / 'confusion_matrix_comparison.png')
    plt.close()

def plot_cm(cm, class_names, title):
    """Plot a specific confusion matrix."""
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

def print_comparison_results(results):
    """Print comparison results in a readable format."""
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    
    # Only proceed if we have results for both models
    if 'team' not in results or 'attention' not in results:
        print("Cannot print comparison - missing model results")
        return
    
    # Calculate overall accuracy
    team_acc = results['team']['accuracy']
    attention_acc = results['attention']['accuracy']
    better_model = "Team-Based" if team_acc > attention_acc else "Attention-Based"
    accuracy_diff = abs(team_acc - attention_acc) * 100
    
    print(f"\nOverall Accuracy:")
    print(f"  Team-Based Predictor: {team_acc:.4f} ({team_acc*100:.2f}%)")
    print(f"  Attention-Based Predictor: {attention_acc:.4f} ({attention_acc*100:.2f}%)")
    print(f"  The {better_model} Predictor is better by {accuracy_diff:.2f}%")
    
    # Calculate inference time
    team_time = results['team']['inference_time']
    attention_time = results['attention']['inference_time']
    faster_model = "Team-Based" if team_time < attention_time else "Attention-Based"
    time_ratio = max(team_time, attention_time) / min(team_time, attention_time)
    
    print(f"\nInference Time:")
    print(f"  Team-Based Predictor: {team_time:.6f} seconds")
    print(f"  Attention-Based Predictor: {attention_time:.6f} seconds")
    print(f"  The {faster_model} Predictor is {time_ratio:.2f}x faster")
    
    # Print class-wise metrics
    print("\nPer-Class Metrics:")
    class_names = ['Away Win', 'Draw', 'Home Win']
    
    for class_name in class_names:
        print(f"\n  {class_name}:")
        
        team_metrics = next(m for m in results['team']['class_metrics'] if m['class'] == class_name)
        attention_metrics = next(m for m in results['attention']['class_metrics'] if m['class'] == class_name)
        
        metrics_to_show = {
            'Accuracy': 'accuracy',
            'Precision': 'precision',
            'Recall': 'recall',
            'F1 Score': 'f1'
        }
        
        for metric_name, metric_key in metrics_to_show.items():
            team_value = team_metrics[metric_key]
            attention_value = attention_metrics[metric_key]
            better = "Team-Based" if team_value > attention_value else "Attention-Based"
            
            print(f"    {metric_name}:")
            print(f"      Team-Based: {team_value:.4f}")
            print(f"      Attention-Based: {attention_value:.4f}")
            print(f"      Better: {better} Predictor")
    
    # Print summary
    print("\nSUMMARY:")
    
    # Determine overall better model based on accuracy
    if team_acc > attention_acc:
        print(f"  The Team-Based Predictor has better overall accuracy (+{accuracy_diff:.2f}%)")
    else:
        print(f"  The Attention-Based Predictor has better overall accuracy (+{accuracy_diff:.2f}%)")
    
    # Determine better model per class
    better_by_class = {'Team-Based': 0, 'Attention-Based': 0}
    
    for class_name in class_names:
        team_metrics = next(m for m in results['team']['class_metrics'] if m['class'] == class_name)
        attention_metrics = next(m for m in results['attention']['class_metrics'] if m['class'] == class_name)
        
        team_f1 = team_metrics['f1']
        attention_f1 = attention_metrics['f1']
        better = "Team-Based" if team_f1 > attention_f1 else "Attention-Based"
        better_by_class[better] += 1
    
    print(f"  Classes better predicted:")
    print(f"    Team-Based Predictor: {better_by_class['Team-Based']} classes")
    print(f"    Attention-Based Predictor: {better_by_class['Attention-Based']} classes")
    
    # Overall recommendation
    print("\nRECOMMENDATION:")
    if team_acc > attention_acc and better_by_class['Team-Based'] >= better_by_class['Attention-Based']:
        print("  Use the Team-Based Predictor for better accuracy and class-specific performance")
    elif attention_acc > team_acc and better_by_class['Attention-Based'] >= better_by_class['Team-Based']:
        print("  Use the Attention-Based Predictor for better accuracy and class-specific performance")
    else:
        print("  Consider using both models in an ensemble for optimal results")
    
    print("="*50)

def main():
    """Main function for model comparison."""
    print("\n========== Football Match Prediction Model Comparison ==========")
    
    # Load data
    print("\nLoading team data...")
    X, y, feature_columns = load_team_data()
    
    # Prepare train/test split
    print("Preparing data for evaluation...")
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(X, y)
    print(f"Test set size: {len(X_test)} samples")
    
    # Load models
    print("\nLoading models...")
    team_model, attention_model = load_models()
    
    # Compare models if both are available
    if team_model and attention_model:
        print("\nComparing model performance...")
        results = compare_models(team_model, attention_model, X_test, y_test)
        
        # Print detailed comparison
        print_comparison_results(results)
        
        # Plot comparison metrics
        print("\nGenerating comparison visualizations...")
        plot_comparison_metrics(results)
        print(f"Visualizations saved to {Path(__file__).parent.parent.parent / 'visualizations'}")
    else:
        print("\nCannot compare models - one or both models are missing.")
        print("Please train both models first.")
    
    print("\n=============================================================")

if __name__ == "__main__":
    main() 
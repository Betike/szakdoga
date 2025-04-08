import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
from pathlib import Path
import sys
import joblib
import xgboost as xgb

sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_team_data import load_team_data, prepare_data_for_training
from models.team_based_predictor import TeamBasedPredictor
from models.random_forest_predictor import RandomForestMatchPredictor
from models.xgboost_predictor import XGBoostMatchPredictor

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir='visualizations'):
    """Plot confusion matrix for a model."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Away Win', 'Draw', 'Home Win'],
                yticklabels=['Away Win', 'Draw', 'Home Win'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.close()

def plot_roc_curves(y_true, y_pred_proba, model_name, save_dir='visualizations'):
    """Plot ROC curves for each class."""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = 3
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_roc_curves.png'))
    plt.close()

def plot_feature_importance(model, feature_names, model_name, save_dir='visualizations'):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_booster'):
        importances = model.get_booster().get_score(importance_type='gain')
    else:
        return
    
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 6))
    plt.title(f'Feature Importance - {model_name}')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.png'))
    plt.close()

def evaluate_model(model, X_test, y_test, model_name, feature_names=None):
    """Evaluate a model and generate visualizations."""
    print(f"\nEvaluating {model_name}...")
    
    if isinstance(model, TeamBasedPredictor):
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            outputs = model(X_test_tensor)
            y_pred = torch.max(outputs.data, 1)[1].numpy()
            y_pred_proba = torch.softmax(outputs, dim=1).numpy()
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Away Win', 'Draw', 'Home Win']))
    
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curves(y_test, y_pred_proba, model_name)
    
    if feature_names is not None:
        plot_feature_importance(model, feature_names, model_name)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def compare_models():
    """Compare all models and generate visualizations."""
    os.makedirs('visualizations', exist_ok=True)
    
    print("Loading data...")
    X, y, feature_columns = load_team_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(X, y)
    
    models = {
        'Neural Network': TeamBasedPredictor(input_size=len(feature_columns)),
        'Random Forest': RandomForestMatchPredictor(),
        'XGBoost': XGBoostMatchPredictor()
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if isinstance(model, TeamBasedPredictor):
            train_losses, test_losses, train_accuracies, test_accuracies = model.train_model(
                X_train, y_train, X_test, y_test
            )
        else:
            model.train(X_train, y_train)
        
        results[name] = evaluate_model(model, X_test, y_test, name, feature_columns)
    
    plt.figure(figsize=(10, 6))
    accuracies = [results[model]['accuracy'] for model in models.keys()]
    plt.bar(models.keys(), accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('visualizations/model_accuracy_comparison.png')
    plt.close()
    
    print("\nModel Comparison Summary:")
    print("-" * 50)
    for name, result in results.items():
        print(f"{name}:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    compare_models() 
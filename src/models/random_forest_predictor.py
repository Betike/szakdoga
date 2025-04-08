import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_match_data import load_match_data, prepare_data_for_training

class RandomForestMatchPredictor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.feature_columns = None
        self.scaler = None
    
    def train(self, X_train, y_train, feature_columns, scaler):
        """Train the Random Forest model."""
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Make predictions on new data."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance rankings."""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        return [(self.feature_columns[i], importance[i]) for i in indices]
    
    def save_model(self, path):
        """Save the model and related data."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path):
        """Load a saved model."""
        data = joblib.load(path)
        predictor = cls()
        predictor.model = data['model']
        predictor.feature_columns = data['feature_columns']
        predictor.scaler = data['scaler']
        print(f"Model loaded from {path}")
        return predictor

def plot_feature_importance(predictor):
    """Plot feature importance rankings."""
    importance = predictor.get_feature_importance()
    features, scores = zip(*importance)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(features)), scores)
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Away Win', 'Draw', 'Home Win'],
                yticklabels=['Away Win', 'Draw', 'Home Win'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def evaluate_model(predictor, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = predictor.predict(X_test)
    
    print("\nModel Evaluation:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Away Win', 'Draw', 'Home Win']))
    
    plot_confusion_matrix(y_test, y_pred)
    
    plot_feature_importance(predictor)

def main():
    os.makedirs('models', exist_ok=True)
    
    print("Loading data...")
    df, feature_columns = load_match_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(df, feature_columns)
    
    print("\nTraining Random Forest model...")
    predictor = RandomForestMatchPredictor(n_estimators=200, max_depth=10)
    predictor.train(X_train, y_train, feature_columns, scaler)
    
    cv_scores = cross_val_score(predictor.model, X_train, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    evaluate_model(predictor, X_test, y_test)
    
    predictor.save_model('models/random_forest_predictor.joblib')
    
    print("\nLoading saved model...")
    loaded_predictor = RandomForestMatchPredictor.load_model('models/random_forest_predictor.joblib')
    evaluate_model(loaded_predictor, X_test, y_test)

if __name__ == "__main__":
    main() 
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_match_data import load_match_data, prepare_data_for_training

class XGBoostMatchPredictor:
    def __init__(self, learning_rate=0.1, max_depth=5, n_estimators=100, random_state=42):
        self.model = xgb.XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=random_state,
            eval_metric='mlogloss',
            use_label_encoder=False,
            objective='multi:softprob',
            num_class=3,
            tree_method='hist'  # Faster training method
        )
        self.feature_columns = None
        self.scaler = None
    
    def train(self, X_train, y_train, feature_columns, scaler, eval_set=None):
        """Train the XGBoost model."""
        self.feature_columns = feature_columns
        self.scaler = scaler
        
        # Calculate class weights
        classes, counts = np.unique(y_train, return_counts=True)
        weight_dict = {c: len(y_train) / (len(classes) * count) for c, count in zip(classes, counts)}
        sample_weights = np.array([weight_dict[y] for y in y_train])
        
        self.model.fit(
            X_train, 
            y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=True
        )
    
    def tune_hyperparameters(self, X_train, y_train, cv=3):
        """Tune hyperparameters using GridSearchCV."""
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5]
        }
        
        # Create a smaller grid for faster tuning
        small_param_grid = {
            'learning_rate': [0.1],
            'max_depth': [3, 5],
            'n_estimators': [100],
            'gamma': [0, 0.1],
            'min_child_weight': [1, 3]
        }
        
        # Set up CV with stratification
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Run GridSearchCV
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=small_param_grid,  # Use small grid for faster results
            scoring='accuracy',
            cv=cv_split,
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = xgb.XGBClassifier(
            **grid_search.best_params_,
            use_label_encoder=False,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss'
        )
        
        return grid_search.best_params_
    
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
    plt.savefig('xgboost_feature_importance.png')
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
    plt.savefig('xgboost_confusion_matrix.png')
    plt.close()

def evaluate_model(predictor, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = predictor.predict(X_test)
    y_proba = predictor.predict_proba(X_test)
    
    print("\nModel Evaluation:")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Away Win', 'Draw', 'Home Win']))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot feature importance
    plot_feature_importance(predictor)
    
    # Plot probability distributions
    plot_probability_distributions(y_test, y_proba)

def plot_probability_distributions(y_true, y_proba):
    """Plot probability distributions for each class."""
    class_names = ['Away Win', 'Draw', 'Home Win']
    plt.figure(figsize=(15, 5))
    
    for i, class_name in enumerate(class_names):
        plt.subplot(1, 3, i+1)
        
        # Get probabilities for true positives and true negatives
        true_mask = (y_true == i)
        probs_true = y_proba[true_mask, i]
        probs_false = y_proba[~true_mask, i]
        
        # Plot distributions
        sns.histplot(probs_true, color='green', alpha=0.5, bins=20, 
                     kde=True, stat='density', label=f'True {class_name}')
        sns.histplot(probs_false, color='red', alpha=0.5, bins=20, 
                     kde=True, stat='density', label=f'Not {class_name}')
        
        plt.title(f'{class_name} Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('xgboost_probability_distributions.png')
    plt.close()

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and prepare data
    print("Loading data...")
    df, feature_columns = load_match_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(df, feature_columns)
    
    # Initialize XGBoost model
    predictor = XGBoostMatchPredictor()
    
    # Tune hyperparameters (optional - can be time consuming)
    tune_params = False
    if tune_params:
        print("\nTuning hyperparameters...")
        best_params = predictor.tune_hyperparameters(X_train, y_train)
    
    # Train model
    print("\nTraining XGBoost model...")
    predictor.train(
        X_train, 
        y_train, 
        feature_columns, 
        scaler,
        eval_set=[(X_test, y_test)]
    )
    
    # Evaluate model
    evaluate_model(predictor, X_test, y_test)
    
    # Save model
    predictor.save_model('models/xgboost_predictor.joblib')
    
    # Example of loading and using the saved model
    print("\nLoading saved model...")
    loaded_predictor = XGBoostMatchPredictor.load_model('models/xgboost_predictor.joblib')
    evaluate_model(loaded_predictor, X_test, y_test)

if __name__ == "__main__":
    main() 
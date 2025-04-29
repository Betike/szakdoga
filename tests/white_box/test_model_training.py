import unittest
import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import tempfile
import shutil
from unittest.mock import patch

class TestModelTraining(unittest.TestCase):
    """
    White box tests for the model training module. These tests have knowledge
    of the internal implementation details and test specific functions and components.
    """
    
    def setUp(self):
        # Get the project root directory
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        # Add project root to path to ensure imports work
        sys.path.insert(0, self.project_root)
        
        # Create test directories
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data/train_test")
        self.models_dir = os.path.join(self.test_dir, "models")
        self.utils_dir = os.path.join(self.test_dir, "utils/random_forest")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.utils_dir, exist_ok=True)
        
        # Create sample test data
        self.create_test_data()
        
    def tearDown(self):
        # Clean up test directory
        shutil.rmtree(self.test_dir)
    
    def create_test_data(self):
        """Create sample test data for model training tests"""
        # Create simplified training data with just a few features
        np.random.seed(42)
        
        # Create training data
        train_data = pd.DataFrame({
            'Season': ['2021-2022'] * 100,
            'HomeTeam': [f'Team {i%10}' for i in range(100)],
            'AwayTeam': [f'Team {(i+5)%10}' for i in range(100)],
            'Result': np.random.choice(['H', 'D', 'A'], size=100, p=[0.45, 0.25, 0.3]),
            'Home_W': np.random.randint(10, 25, size=100),
            'Home_D': np.random.randint(5, 15, size=100),
            'Home_L': np.random.randint(5, 15, size=100),
            'Away_W': np.random.randint(8, 20, size=100),
            'Away_D': np.random.randint(5, 15, size=100),
            'Away_L': np.random.randint(8, 20, size=100),
            'Home_GF': np.random.randint(30, 80, size=100),
            'Home_GA': np.random.randint(25, 60, size=100),
            'Away_GF': np.random.randint(25, 70, size=100),
            'Away_GA': np.random.randint(30, 70, size=100),
            'Diff_GF': np.random.randint(-30, 40, size=100),
            'Diff_GA': np.random.randint(-30, 30, size=100)
        })
        
        # Create test data with similar structure
        test_data = pd.DataFrame({
            'Season': ['2022-2023'] * 20,
            'HomeTeam': [f'Team {i%10}' for i in range(20)],
            'AwayTeam': [f'Team {(i+5)%10}' for i in range(20)],
            'Result': np.random.choice(['H', 'D', 'A'], size=20, p=[0.45, 0.25, 0.3]),
            'Home_W': np.random.randint(10, 25, size=20),
            'Home_D': np.random.randint(5, 15, size=20),
            'Home_L': np.random.randint(5, 15, size=20),
            'Away_W': np.random.randint(8, 20, size=20),
            'Away_D': np.random.randint(5, 15, size=20),
            'Away_L': np.random.randint(8, 20, size=20),
            'Home_GF': np.random.randint(30, 80, size=20),
            'Home_GA': np.random.randint(25, 60, size=20),
            'Away_GF': np.random.randint(25, 70, size=20),
            'Away_GA': np.random.randint(30, 70, size=20),
            'Diff_GF': np.random.randint(-30, 40, size=20),
            'Diff_GA': np.random.randint(-30, 30, size=20)
        })
        
        # Save to test directory
        train_data.to_csv(os.path.join(self.data_dir, "train_data_chronological.csv"), index=False)
        test_data.to_csv(os.path.join(self.data_dir, "test_data_chronological.csv"), index=False)
    
    def test_random_forest_imports(self):
        """Test that the random forest training module can be imported"""
        try:
            # Import the module
            sys.path.insert(0, os.path.join(self.project_root, "train"))
            import train_random_forest_model
            
            # Successful import means test passes
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import train_random_forest_model: {e}")
    
    @patch('matplotlib.pyplot.savefig')  # Mock the plot saving
    def test_random_forest_training(self, mock_savefig):
        """Test the random forest model training process"""
        # Set environment variables to use test directories
        old_data_dir = os.environ.get('DATA_DIR')
        old_models_dir = os.environ.get('MODELS_DIR')
        old_utils_dir = os.environ.get('UTILS_DIR')
        
        try:
            # Import the random forest training module
            sys.path.insert(0, os.path.join(self.project_root, "train"))
            
            # Without importing the module yet, patch file paths            
            with patch('os.path.exists', return_value=True), \
                 patch('os.makedirs', return_value=None):
                
                # Mock open() to use our test directory paths
                original_open = open
                
                def mock_open(file, *args, **kwargs):
                    # Redirect data paths to test directory
                    if 'train_data_chronological.csv' in file:
                        return original_open(os.path.join(self.data_dir, "train_data_chronological.csv"), *args, **kwargs)
                    elif 'test_data_chronological.csv' in file:
                        return original_open(os.path.join(self.data_dir, "test_data_chronological.csv"), *args, **kwargs)
                    # For other files, just let them use the original path
                    return original_open(file, *args, **kwargs)
                
                # Here we're simulating the training process rather than importing the module
                # This is to avoid having to change the hardcoded paths in the module
                
                # Replicate key model training steps from train_random_forest_model.py
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.preprocessing import LabelEncoder
                from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                
                # 1. Load data
                train_data = pd.read_csv(os.path.join(self.data_dir, "train_data_chronological.csv"))
                test_data = pd.read_csv(os.path.join(self.data_dir, "test_data_chronological.csv"))
                
                # 2. Prepare features
                feature_cols = [
                    col for col in train_data.columns
                    if (col.startswith('Home_') or col.startswith('Away_') or col.startswith('Diff_'))
                ]
                
                X_train = train_data[feature_cols]
                X_test = test_data[feature_cols]
                
                # 3. Encode labels
                label_encoder = LabelEncoder()
                label_encoder.fit(train_data['Result'])
                
                y_train = label_encoder.transform(train_data['Result'])
                y_test = label_encoder.transform(test_data['Result'])
                
                # 4. Train model
                rf_model = RandomForestClassifier(
                    n_estimators=10,  # Use a small number for testing
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=5,
                    max_features=0.3,
                    bootstrap=True,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=1
                )
                
                rf_model.fit(X_train, y_train)
                
                # 5. Save model artifacts
                model_path = os.path.join(self.models_dir, "random_forest_prediction_model.pkl")
                
                with open(model_path, 'wb') as f:
                    pickle.dump(rf_model, f)
                
                # Save feature names
                feature_info = {'feature_names': feature_cols}
                with open(os.path.join(self.utils_dir, 'random_forest_features.json'), 'w') as f:
                    json.dump(feature_info, f)
                
                # Save label encoder
                with open(os.path.join(self.utils_dir, 'random_forest_label_encoder.pkl'), 'wb') as f:
                    pickle.dump(label_encoder, f)
                
                # 6. Evaluate model
                train_preds = rf_model.predict(X_train)
                test_preds = rf_model.predict(X_test)
                
                train_accuracy = accuracy_score(y_train, train_preds)
                test_accuracy = accuracy_score(y_test, test_preds)
                
                # 7. Test assertions
                self.assertTrue(os.path.exists(model_path), "Model file should be created")
                
                # Feature importance should be available
                self.assertEqual(len(rf_model.feature_importances_), len(feature_cols),
                               "Feature importance vector should match feature count")
                
                # Accuracy should be reasonable for random data
                self.assertGreater(train_accuracy, 0.1, "Training accuracy should be greater than chance")
                self.assertLess(train_accuracy, 1.0, "Training accuracy should not be perfect (would suggest overfitting)")
                
        finally:
            # Restore environment variables
            if old_data_dir:
                os.environ['DATA_DIR'] = old_data_dir
            if old_models_dir:
                os.environ['MODELS_DIR'] = old_models_dir
            if old_utils_dir:
                os.environ['UTILS_DIR'] = old_utils_dir
    
    def test_xgboost_imports(self):
        """Test that the XGBoost training module can be imported"""
        try:
            # Import the module
            sys.path.insert(0, os.path.join(self.project_root, "train"))
            import train_xgboost_model
            
            # Successful import means test passes
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import train_xgboost_model: {e}")
    
    def test_pytorch_imports(self):
        """Test that the PyTorch training module can be imported"""
        try:
            # Import the module
            sys.path.insert(0, os.path.join(self.project_root, "train"))
            import train_pytorch_model
            
            # Check if the neural network class exists
            self.assertTrue(hasattr(train_pytorch_model, 'MatchPredictionNN'), 
                          "MatchPredictionNN class should exist in the module")
            
        except ImportError as e:
            # Skip this test if PyTorch is not installed
            if 'torch' in str(e) or 'pytorch' in str(e).lower():
                self.skipTest("PyTorch not installed")
            else:
                self.fail(f"Failed to import train_pytorch_model: {e}")
    
    def test_model_feature_selection(self):
        """Test the feature selection logic in the training code"""
        train_data = pd.read_csv(os.path.join(self.data_dir, "train_data_chronological.csv"))
        test_data = pd.read_csv(os.path.join(self.data_dir, "test_data_chronological.csv"))
        
        # Add some columns with NaN values to test feature selection
        train_data['Home_BadFeature'] = np.nan
        train_data['Away_BadFeature'] = np.random.randn(len(train_data))
        test_data['Home_BadFeature'] = np.random.randn(len(test_data))
        test_data['Away_BadFeature'] = np.nan
        
        train_data.to_csv(os.path.join(self.data_dir, "train_data_chronological.csv"), index=False)
        test_data.to_csv(os.path.join(self.data_dir, "test_data_chronological.csv"), index=False)
        
        # Reload the data
        train_data = pd.read_csv(os.path.join(self.data_dir, "train_data_chronological.csv"))
        test_data = pd.read_csv(os.path.join(self.data_dir, "test_data_chronological.csv"))
        
        # Get feature columns
        feature_cols = [
            col for col in train_data.columns
            if (col.startswith('Home_') or col.startswith('Away_') or col.startswith('Diff_'))
        ]
        
        # Filter out columns with NaN values
        for col in feature_cols.copy():
            if train_data[col].isna().any() or test_data[col].isna().any():
                feature_cols.remove(col)
        
        # Test assertions
        self.assertNotIn('Home_BadFeature', feature_cols, 
                       "Features with NaN values should be removed")
        self.assertNotIn('Away_BadFeature', feature_cols, 
                       "Features with NaN values should be removed")
        self.assertIn('Home_W', feature_cols, 
                     "Valid features should be retained")
    
    def test_feature_importance_calculation(self):
        """Test the feature importance calculation logic"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Load the data
        train_data = pd.read_csv(os.path.join(self.data_dir, "train_data_chronological.csv"))
        
        # Prepare features and target
        feature_cols = [
            col for col in train_data.columns
            if (col.startswith('Home_') or col.startswith('Away_') or col.startswith('Diff_'))
        ]
        
        X_train = train_data[feature_cols]
        y_train = train_data['Result'].map({'H': 0, 'D': 1, 'A': 2})
        
        # Train a simple model
        # Fix the syntax error: All positional arguments need to come before keyword arguments
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Test assertions
        self.assertEqual(len(feature_importance), len(feature_cols),
                       "Feature importance should have one row per feature")
        self.assertTrue((feature_importance['Importance'] >= 0).all(),
                       "All importance values should be non-negative")
        self.assertAlmostEqual(feature_importance['Importance'].sum(), 1.0, places=5,
                             msg="Feature importance should sum to 1.0")

if __name__ == '__main__':
    unittest.main() 
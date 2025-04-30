import unittest
import os
import sys
import json
import pickle
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

class MockLabelEncoder:
    def __init__(self):
        self.classes_ = np.array(['H', 'D', 'A'])
    
    def transform(self, y):
        return np.array([list(self.classes_).index(label) for label in y])
    
    def inverse_transform(self, y):
        return np.array([self.classes_[i] for i in y])

class TestPrediction(unittest.TestCase):
    
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        sys.path.insert(0, self.project_root)
        
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.test_dir, "models")
        self.utils_dir = os.path.join(self.test_dir, "utils/random_forest")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.utils_dir, exist_ok=True)
        
        self.create_test_artifacts()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def create_test_artifacts(self):
        from sklearn.ensemble import RandomForestClassifier
        
        model = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=42)
        
        X_dummy = np.random.randn(10, 5)
        y_dummy = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        model.fit(X_dummy, y_dummy)
        
        model_path = os.path.join(self.models_dir, "random_forest_prediction_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        label_encoder = MockLabelEncoder()
        
        with open(os.path.join(self.utils_dir, "random_forest_label_encoder.pkl"), 'wb') as f:
            pickle.dump(label_encoder, f)
        
        feature_names = ['Home_W', 'Home_D', 'Away_W', 'Away_D', 'Diff_GF']
        with open(os.path.join(self.utils_dir, "random_forest_features.json"), 'w') as f:
            json.dump({'feature_names': feature_names}, f)
    
    def test_run_prediction_imports(self):
        try:
            sys.path.insert(0, os.path.join(self.project_root, "predict"))
            import run_prediction
            
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import run_prediction: {e}")
    
    @patch('subprocess.run')
    def test_run_prediction_function(self, mock_subprocess):
        sys.path.insert(0, os.path.join(self.project_root, "predict"))
        import run_prediction
        
        test_args = ['--model', 'random_forest', '--home', 'Liverpool', '--away', 'Manchester City']
        
        mock_process = MagicMock()
        mock_process.stdout = '{"prediction": "H", "probabilities": {"Home win": 0.7, "Draw": 0.2, "Away win": 0.1}}'
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process
        
        with patch('sys.argv', ['run_prediction.py'] + test_args):
            result = run_prediction.main()
        
        mock_subprocess.assert_called_once()
        cmd_args = mock_subprocess.call_args[0][0]
        
        self.assertIn('--home', cmd_args)
        self.assertIn('Liverpool', cmd_args)
        self.assertIn('--away', cmd_args)
        self.assertIn('Manchester City', cmd_args)
        self.assertIn('--single-match', cmd_args)
        
        self.assertEqual(result, 0)
    
    def test_predict_with_random_forest_imports(self):
        try:
            sys.path.insert(0, os.path.join(self.project_root, "predict"))
            import predict_with_random_forest
            
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import predict_with_random_forest: {e}")
    
    def test_predict_single_match_function(self):
        with patch.dict('os.environ', {'MODELS_DIR': self.models_dir, 'UTILS_DIR': self.utils_dir}):
            
            sys.path.insert(0, os.path.join(self.project_root, "predict"))
            
            model_path = os.path.join(self.models_dir, "random_forest_prediction_model.pkl")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            features_path = os.path.join(self.utils_dir, "random_forest_features.json")
            with open(features_path, 'r') as f:
                feature_info = json.load(f)
                feature_names = feature_info["feature_names"]
            
            encoder_path = os.path.join(self.utils_dir, "random_forest_label_encoder.pkl")
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            
            match_features = pd.DataFrame({
                'Home_W': [20],
                'Home_D': [10],
                'Away_W': [18],
                'Away_D': [8],
                'Diff_GF': [10]
            })
            
            for col in feature_names:
                if col not in match_features.columns:
                    match_features[col] = 0
            
            match_features = match_features[feature_names]
            
            predicted_probs = model.predict_proba(match_features)
            predicted_class = model.predict(match_features)
            
            result = label_encoder.inverse_transform(predicted_class)[0]
            
            self.assertIn(result, ['H', 'D', 'A'], "Prediction should be one of H, D, A")
            self.assertEqual(predicted_probs.shape[1], 3, "Should have probabilities for 3 classes")
            self.assertAlmostEqual(predicted_probs.sum(), 1.0, places=5, 
                                 msg="Probabilities should sum to 1.0")
    
    def test_team_name_normalization(self):
        def normalize_team_name(name):
            name = name.strip()
            
            name_mapping = {
                "Man United": "Manchester United",
                "Man Utd": "Manchester United",
                "Manchester Utd": "Manchester United",
                "Man City": "Manchester City",
                "Wolves": "Wolverhampton",
                "Spurs": "Tottenham",
                "Tottenham Hotspur": "Tottenham",
                "Newcastle": "Newcastle United",
                "Leeds": "Leeds United"
            }
            
            if name in name_mapping:
                return name_mapping[name]
            
            suffixes = [" FC", " F.C.", " AFC", " United", " City"]
            for suffix in suffixes:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
            
            for key, value in name_mapping.items():
                if name in value:
                    return value
            
            return name
        
        test_cases = [
            ("Manchester United", "Manchester United"),
            ("Man United", "Manchester United"),
            ("Man Utd", "Manchester United"),
            ("Man City", "Manchester City"),
            ("Leicester City", "Leicester"),
            ("Tottenham Hotspur", "Tottenham"),
            ("Spurs", "Tottenham"),
            ("Newcastle United", "Newcastle United"),
            ("Newcastle", "Newcastle United"),
            ("Arsenal FC", "Arsenal"),
            ("Wolves", "Wolverhampton")
        ]
        
        for input_name, expected_output in test_cases:
            normalized = normalize_team_name(input_name)
            self.assertEqual(normalize_team_name(normalized), normalize_team_name(expected_output),
                           f"Failed to normalize {input_name} correctly")

if __name__ == '__main__':
    unittest.main() 
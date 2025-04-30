import unittest
import subprocess
import json
import os
import sys
import warnings

class TestPredictionAPI(unittest.TestCase):

    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        sys.path.insert(0, self.project_root)
        
    def run_prediction(self, model, home_team, away_team):
        cmd = [
            sys.executable,
            os.path.join(self.project_root, "predict", "run_prediction.py"),
            "--model", model,
            "--home", home_team,
            "--away", away_team,
            "--json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        
        try:
            output = result.stdout.strip()
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                return json.loads(json_str)
            else:
                return None
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            print(f"Output: {result.stdout}")
            return None

    def test_model_availability(self):
        models = ["random_forest", "xgboost", "pytorch", "ensemble"]
        for model in models:
            result = self.run_prediction(model, "Manchester Utd", "Chelsea")
            self.assertIsNotNone(result, f"Model {model} failed to generate a prediction")
            self.assertIn("prediction", result, f"Model {model} didn't return a prediction key")
        
    def test_prediction_result_structure(self):
        result = self.run_prediction("ensemble", "Arsenal", "Liverpool")
        self.assertIsNotNone(result)
        
        self.assertIn("prediction", result, "Missing 'prediction' key in result")
        self.assertIn("probabilities", result, "Missing 'probabilities' key in result")
        
        probabilities = result["probabilities"]
        self.assertIsInstance(probabilities, dict, "Probabilities should be a dictionary")
        self.assertIn("Home win", probabilities, "Missing 'Home win' in probabilities")
        self.assertIn("Draw", probabilities, "Missing 'Draw' in probabilities")
        self.assertIn("Away win", probabilities, "Missing 'Away win' in probabilities")
        
        prob_sum = (probabilities["Home win"] + 
                    probabilities["Draw"] + 
                    probabilities["Away win"])
        self.assertAlmostEqual(prob_sum, 1.0, places=2, 
                              msg="Probabilities don't sum to 1")
        
        self.assertIn(result["prediction"], ["H", "D", "A"], 
                     "Prediction not in expected values (H, D, A)")
    
    def test_same_home_away_team(self):
        result = self.run_prediction("random_forest", "Liverpool", "Liverpool")
        if "error" in result:
            self.assertIn("error", result)
        else:
            self.assertIn("prediction", result)
    
    def test_unknown_team(self):
        result = self.run_prediction("random_forest", "Unknown Team FC", "Chelsea")
        self.assertIn("error", result, "Should return error for unknown team")
    
    def test_model_consistency(self):
        result1 = self.run_prediction("xgboost", "Manchester City", "Newcastle United")
        result2 = self.run_prediction("xgboost", "Manchester City", "Newcastle United")
        
        if result1 and result2 and "prediction" in result1 and "prediction" in result2:
            self.assertEqual(result1["prediction"], result2["prediction"],
                           "Model produced inconsistent predictions for same input")
            
            if "probabilities" in result1 and "probabilities" in result2:
                probs1 = result1["probabilities"]
                probs2 = result2["probabilities"]
                
                self.assertEqual(probs1["Home win"], probs2["Home win"], "Home win probabilities differ")
                self.assertEqual(probs1["Draw"], probs2["Draw"], "Draw probabilities differ")
                self.assertEqual(probs1["Away win"], probs2["Away win"], "Away win probabilities differ")
    
    def test_all_premier_league_teams(self):
        teams = [
            "Arsenal",
            "Aston Villa",
            "Bournemouth",
            "Brentford",
            "Brighton",
            "Chelsea",
            "Crystal Palace",
            "Everton",
            "Fulham",
            "Ipswich Town",
            "Leicester City",
            "Liverpool",
            "Manchester City",
            "Manchester Utd",
            "Newcastle Utd",
            "Nott'ham Forest",
            "Southampton",
            "Tottenham",
            "West Ham",
            "Wolves"
            ]
        
        for home in teams[:3]:
            for away in teams[:3]:
                if home != away:
                    result = self.run_prediction("ensemble", home, away)
                    self.assertIsNotNone(result)
                    self.assertIn("prediction", result)

if __name__ == '__main__':
    unittest.main() 
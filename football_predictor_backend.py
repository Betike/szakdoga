import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

def setup_environment():
    # Beállítja a környezetet, hogy a csomagolt verzióban is működjön
    if getattr(sys, 'frozen', False):
        # PyInstaller által csomagolt verzió
        base_path = sys._MEIPASS
    else:
        # Normál Python futtatás
        base_path = os.path.abspath(os.path.dirname(__file__))
    
    os.environ['MODEL_PATH'] = os.path.join(base_path, 'models')
    os.environ['DATA_PATH'] = os.path.join(base_path, 'data')
    
    # Biztosítsuk, hogy a szükséges mappák léteznek
    os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'data'), exist_ok=True)
    
    return base_path

def run_prediction(home_team, away_team, model_type='ensemble'):
    """Predikciót futtat a megadott csapatokra és modell típusra."""
    try:
        cmd = [
            'python', 
            os.path.join('predict', 'run_prediction.py'),
            '--model', model_type,
            '--home', home_team,
            '--away', away_team,
            '--json'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.stdout:
            # Csak a JSON részt vegyük ki
            json_start = result.stdout.find('{')
            json_end = result.stdout.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result.stdout[json_start:json_end]
                return json.loads(json_str)
        
        return {"error": "No valid prediction result"}
        
    except subprocess.CalledProcessError as e:
        return {"error": f"Prediction failed: {e.stderr}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def start_prediction_server(port=5000):
    """Elindít egy Flask szervert a predikciós API-nak."""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        
        app = Flask(__name__)
        CORS(app)  # Engedélyezi a kereszt-forrású kéréseket
        
        @app.route('/api/predict', methods=['POST'])
        def predict():
            try:
                data = request.json
                if not data:
                    return jsonify({"error": "No data provided"}), 400
                
                home_team = data.get('homeTeam')
                away_team = data.get('awayTeam')
                model_type = data.get('predictorType', 'ensemble')
                
                if not home_team or not away_team:
                    return jsonify({"error": "Home and away teams are required"}), 400
                
                result = run_prediction(home_team, away_team, model_type)
                return jsonify(result)
                
            except Exception as e:
                return jsonify({"error": f"Server error: {str(e)}"}), 500
        
        @app.route('/api/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "ok"})
        
        print(f"Starting prediction server on port {port}...")
        app.run(host='127.0.0.1', port=port)
        
    except Exception as e:
        print(f"Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Football Predictor Backend')
    parser.add_argument('--port', type=int, default=5000, help='Port for the prediction server')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode instead of server')
    parser.add_argument('--home', type=str, help='Home team (CLI mode only)')
    parser.add_argument('--away', type=str, help='Away team (CLI mode only)')
    parser.add_argument('--model', type=str, default='ensemble', help='Model type (CLI mode only)')
    args = parser.parse_args()
    
    base_path = setup_environment()
    print(f"Using base path: {base_path}")
    
    if args.cli:
        # CLI mód a közvetlen predikciókhoz
        if not args.home or not args.away:
            print("Error: Home and away teams are required in CLI mode")
            parser.print_help()
            sys.exit(1)
            
        result = run_prediction(args.home, args.away, args.model)
        print(json.dumps(result, indent=2))
    else:
        # Szerver mód a frontend-nek
        start_prediction_server(args.port)

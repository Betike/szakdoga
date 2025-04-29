import pandas as pd
import numpy as np
import pickle
import torch
import xgboost as xgb
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime


def main():
    # Create output directory
    os.makedirs('compare/results', exist_ok=True)
    
    # 1. Load test data
    print("Loading test dataset...")
    test_data = pd.read_csv("data/train_test/test_data_chronological.csv")
    print(f"Test data shape: {test_data.shape}")
    
    # Extract the target values
    y_true = test_data['Result']

    # Initialize variables for model predictions
    pytorch_predictions = None
    xgboost_predictions = None
    randomforest_predictions = None

    # Prepare for storing results
    model_results = {
        'Model': [],
        'Accuracy': [],
        'F1_Score_Macro': [],
        'F1_Score_Weighted': [],
        'Home_Win_Precision': [],
        'Home_Win_Recall': [],
        'Draw_Precision': [],
        'Draw_Recall': [],
        'Away_Win_Precision': [],
        'Away_Win_Recall': []
    }

    # Define function to load and evaluate a model
    def evaluate_model(model_type):
        print(f"\nEvaluating {model_type} model...")
        
        if model_type == 'PyTorch':
            # Load PyTorch model
            try:
                # Try different import approaches for pytorch model
                try:
                    # First try the relative import (when run as python -m compare.compare_models)
                    from train.train_pytorch_model import MatchPredictionNN
                except ImportError:
                    try:
                        # Then try adjusting path and using absolute import
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        if project_root not in sys.path:
                            sys.path.insert(0, project_root)
                        from train.train_pytorch_model import MatchPredictionNN
                    except ImportError:
                        # Last resort - direct import when script is run from root
                        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                        from train.train_pytorch_model import MatchPredictionNN
                
                # 2. Load feature scaling information
                with open('utils/neural_network/feature_scaling_info.json', 'r') as f:
                    feature_info = json.load(f)
                
                feature_names = feature_info['feature_names']
                scaler = StandardScaler()
                scaler.mean_ = np.array(feature_info['scaler_mean'])
                scaler.scale_ = np.array(feature_info['scaler_scale'])
                # Set feature_names_in_ attribute to avoid warning
                scaler.feature_names_in_ = np.array(feature_names, dtype=object)
                scaler.n_features_in_ = len(feature_names)
                
                # 3. Load model weights
                model_path = "models/pytorch_model.pth"
                if not os.path.exists(model_path):
                    print(f"PyTorch model not found at {model_path}. Skipping.")
                    return None
                
                # Initialize model with proper input size
                input_size = len(feature_names)
                model = MatchPredictionNN(input_size)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                
                # 4. Prepare features
                X_test = test_data[feature_names]
                X_test_scaled = scaler.transform(X_test)
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                
                # 5. Make predictions
                with torch.no_grad():
                    outputs = model(X_test_tensor)
                    _, predicted = torch.max(outputs, 1)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 6. Convert predictions to original labels
                label_map_inverse = {0: 'A', 1: 'D', 2: 'H'}
                predictions = [label_map_inverse[p.item()] for p in predicted]
                
                # Store probabilities by class
                all_probs = probabilities.cpu().numpy()
                proba_df = pd.DataFrame({
                    'Away_Win_Prob': all_probs[:, 0],
                    'Draw_Prob': all_probs[:, 1],
                    'Home_Win_Prob': all_probs[:, 2]
                })
                
                return predictions, proba_df
                
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")
                import traceback
                traceback.print_exc()
                return None
                
        elif model_type == 'XGBoost':
            # Load XGBoost model
            try:
                # 1. Load model
                model_path = "models/xgboost_prediction_model.json"
                if not os.path.exists(model_path):
                    print(f"XGBoost model not found at {model_path}. Skipping.")
                    return None
                    
                # Load XGBoost model
                model = xgb.Booster()
                model.load_model(model_path)
                
                # 2. Load label encoder
                with open('utils/xgboost/xgboost_label_encoder.pkl', 'rb') as f:
                    label_encoder = pickle.load(f)
                
                # 3. Get feature names
                feature_names = model.feature_names
                if feature_names is None:
                    # If still None, use all numeric features as fallback
                    feature_names = [col for col in test_data.columns 
                                    if (col.startswith('Home_') or col.startswith('Away_') or col.startswith('Diff_')) 
                                    and test_data[col].dtype in [np.int64, np.float64]
                                    and col != 'Home_CrdR' and col != 'Away_CrdR' and col != 'Diff_CrdR']
                
                # 4. Prepare features
                X_test = test_data[feature_names]
                dtest = xgb.DMatrix(X_test)
                
                # 5. Make predictions
                probabilities = model.predict(dtest)
                pred_indices = np.argmax(probabilities, axis=1)
                
                # 6. Convert predictions to original labels
                predictions = label_encoder.inverse_transform(pred_indices)
                
                # Store probabilities by class
                # Get mapping between class index and label
                label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
                
                # Create dataframe with proper column names
                proba_df = pd.DataFrame({
                    'Away_Win_Prob': probabilities[:, list(label_map.keys())[list(label_map.values()).index('A')]],
                    'Draw_Prob': probabilities[:, list(label_map.keys())[list(label_map.values()).index('D')]],
                    'Home_Win_Prob': probabilities[:, list(label_map.keys())[list(label_map.values()).index('H')]]
                })
                
                return predictions, proba_df
                
            except Exception as e:
                print(f"Error loading XGBoost model: {e}")
                import traceback
                traceback.print_exc()
                return None
                
        elif model_type == 'RandomForest':
            # Load Random Forest model
            try:
                # 1. Load model
                model_path = "models/random_forest_prediction_model.pkl"
                if not os.path.exists(model_path):
                    print(f"Random Forest model not found at {model_path}. Skipping.")
                    return None
                    
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # 2. Load label encoder
                with open('utils/random_forest/random_forest_label_encoder.pkl', 'rb') as f:
                    label_encoder = pickle.load(f)
                
                # 3. Get feature names
                feature_path = "utils/random_forest/random_forest_features.json"
                if os.path.exists(feature_path):
                    with open(feature_path, 'r') as f:
                        feature_info = json.load(f)
                        feature_names = feature_info['feature_names']
                else:
                    # Fallback to model's feature_names_ attribute if available
                    try:
                        feature_names = model.feature_names_in_
                    except:
                        # Use a fallback approach
                        feature_names = [col for col in test_data.columns 
                                       if (col.startswith('Home_') or col.startswith('Away_') or col.startswith('Diff_')) 
                                       and test_data[col].dtype in [np.int64, np.float64]
                                       and col != 'Home_CrdR' and col != 'Away_CrdR' and col != 'Diff_CrdR']
                
                # 4. Prepare features
                X_test = test_data[feature_names]
                
                # 5. Make predictions
                probabilities = model.predict_proba(X_test)
                predictions = model.predict(X_test)
                
                # 6. Convert predictions to original labels if needed
                predictions = label_encoder.inverse_transform(predictions)
                
                # Store probabilities by class
                # Get mapping between class index and label
                label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
                
                # Create dataframe with proper column names
                proba_df = pd.DataFrame({
                    'Away_Win_Prob': probabilities[:, list(label_map.keys())[list(label_map.values()).index('A')]],
                    'Draw_Prob': probabilities[:, list(label_map.keys())[list(label_map.values()).index('D')]],
                    'Home_Win_Prob': probabilities[:, list(label_map.keys())[list(label_map.values()).index('H')]]
                })
                
                return predictions, proba_df
                
            except Exception as e:
                print(f"Error loading Random Forest model: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        return None

    # Evaluate each model
    result_pytorch = evaluate_model('PyTorch')
    result_xgboost = evaluate_model('XGBoost')
    result_randomforest = evaluate_model('RandomForest')

    # Create a dataframe of all predictions and the ground truth
    comparison_df = pd.DataFrame({'True_Result': y_true})

    # Add predictions from each model if available
    if result_pytorch is not None:
        pytorch_predictions, pytorch_probs = result_pytorch
        comparison_df['PyTorch_Pred'] = pytorch_predictions
        comparison_df = pd.concat([comparison_df, pytorch_probs.add_prefix('PyTorch_')], axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, pytorch_predictions)
        f1_macro = f1_score(y_true, pytorch_predictions, average='macro')
        f1_weighted = f1_score(y_true, pytorch_predictions, average='weighted')
        
        # Get class-specific metrics
        report = classification_report(y_true, pytorch_predictions, output_dict=True)
        
        # Store results
        model_results['Model'].append('PyTorch')
        model_results['Accuracy'].append(accuracy)
        model_results['F1_Score_Macro'].append(f1_macro)
        model_results['F1_Score_Weighted'].append(f1_weighted)
        model_results['Home_Win_Precision'].append(report['H']['precision'])
        model_results['Home_Win_Recall'].append(report['H']['recall'])
        model_results['Draw_Precision'].append(report['D']['precision'])
        model_results['Draw_Recall'].append(report['D']['recall'])
        model_results['Away_Win_Precision'].append(report['A']['precision'])
        model_results['Away_Win_Recall'].append(report['A']['recall'])
        
        print(f"\nPyTorch Model Accuracy: {accuracy:.4f}")
        print(f"PyTorch Model F1 Score (macro): {f1_macro:.4f}")
        print(f"PyTorch Classification Report:")
        print(classification_report(y_true, pytorch_predictions))

    if result_xgboost is not None:
        xgboost_predictions, xgboost_probs = result_xgboost
        comparison_df['XGBoost_Pred'] = xgboost_predictions
        comparison_df = pd.concat([comparison_df, xgboost_probs.add_prefix('XGBoost_')], axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, xgboost_predictions)
        f1_macro = f1_score(y_true, xgboost_predictions, average='macro')
        f1_weighted = f1_score(y_true, xgboost_predictions, average='weighted')
        
        # Get class-specific metrics
        report = classification_report(y_true, xgboost_predictions, output_dict=True)
        
        # Store results
        model_results['Model'].append('XGBoost')
        model_results['Accuracy'].append(accuracy)
        model_results['F1_Score_Macro'].append(f1_macro)
        model_results['F1_Score_Weighted'].append(f1_weighted)
        model_results['Home_Win_Precision'].append(report['H']['precision'])
        model_results['Home_Win_Recall'].append(report['H']['recall'])
        model_results['Draw_Precision'].append(report['D']['precision'])
        model_results['Draw_Recall'].append(report['D']['recall'])
        model_results['Away_Win_Precision'].append(report['A']['precision'])
        model_results['Away_Win_Recall'].append(report['A']['recall'])
        
        print(f"\nXGBoost Model Accuracy: {accuracy:.4f}")
        print(f"XGBoost Model F1 Score (macro): {f1_macro:.4f}")
        print(f"XGBoost Classification Report:")
        print(classification_report(y_true, xgboost_predictions))

    if result_randomforest is not None:
        randomforest_predictions, randomforest_probs = result_randomforest
        comparison_df['RandomForest_Pred'] = randomforest_predictions
        comparison_df = pd.concat([comparison_df, randomforest_probs.add_prefix('RandomForest_')], axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, randomforest_predictions)
        f1_macro = f1_score(y_true, randomforest_predictions, average='macro')
        f1_weighted = f1_score(y_true, randomforest_predictions, average='weighted')
        
        # Get class-specific metrics
        report = classification_report(y_true, randomforest_predictions, output_dict=True)
        
        # Store results
        model_results['Model'].append('RandomForest')
        model_results['Accuracy'].append(accuracy)
        model_results['F1_Score_Macro'].append(f1_macro)
        model_results['F1_Score_Weighted'].append(f1_weighted)
        model_results['Home_Win_Precision'].append(report['H']['precision'])
        model_results['Home_Win_Recall'].append(report['H']['recall'])
        model_results['Draw_Precision'].append(report['D']['precision'])
        model_results['Draw_Recall'].append(report['D']['recall'])
        model_results['Away_Win_Precision'].append(report['A']['precision'])
        model_results['Away_Win_Recall'].append(report['A']['recall'])
        
        print(f"\nRandom Forest Model Accuracy: {accuracy:.4f}")
        print(f"Random Forest Model F1 Score (macro): {f1_macro:.4f}")
        print(f"Random Forest Classification Report:")
        print(classification_report(y_true, randomforest_predictions))

    # Create a summary of the model results
    results_df = pd.DataFrame(model_results)
    print("\nModel Performance Comparison Summary:")
    print(results_df[['Model', 'Accuracy', 'F1_Score_Macro', 'F1_Score_Weighted']])

    # Save the comparison results
    results_df.to_csv(f'compare/results/model_comparison.csv', index=False)
    comparison_df.to_csv(f'compare/results/prediction_comparison.csv', index=False)

    # Create visualizations

    # 1. Accuracy comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='Accuracy', data=results_df)
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(results_df['Accuracy']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(f'visualisations/comparison/accuracy_comparison.png')

    # 2. F1 Score comparison
    plt.figure(figsize=(12, 8))
    f1_data = results_df.melt(id_vars='Model', 
                             value_vars=['F1_Score_Macro', 'F1_Score_Weighted'],
                             var_name='Metric', value_name='F1 Score')
    sns.barplot(x='Model', y='F1 Score', hue='Metric', data=f1_data)
    plt.title('Model F1 Score Comparison')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'visualisations/comparison/f1_comparison.png')

    # 3. Class-specific metrics
    plt.figure(figsize=(15, 10))

    # Home wins
    plt.subplot(3, 2, 1)
    sns.barplot(x='Model', y='Home_Win_Precision', data=results_df)
    plt.title('Home Win - Precision')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(3, 2, 2)
    sns.barplot(x='Model', y='Home_Win_Recall', data=results_df)
    plt.title('Home Win - Recall')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Draws
    plt.subplot(3, 2, 3)
    sns.barplot(x='Model', y='Draw_Precision', data=results_df)
    plt.title('Draw - Precision')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(3, 2, 4)
    sns.barplot(x='Model', y='Draw_Recall', data=results_df)
    plt.title('Draw - Recall')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Away wins
    plt.subplot(3, 2, 5)
    sns.barplot(x='Model', y='Away_Win_Precision', data=results_df)
    plt.title('Away Win - Precision')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(3, 2, 6)
    sns.barplot(x='Model', y='Away_Win_Recall', data=results_df)
    plt.title('Away Win - Recall')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'visualisations/comparison/class_metrics_comparison.png')

    # 4. Agreement between models
    if len(model_results['Model']) > 1:
        agreement_counts = {}
        
        # Calculate pairwise agreement
        models = [m for m in ['PyTorch_Pred', 'XGBoost_Pred', 'RandomForest_Pred'] if m in comparison_df.columns]
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                agreement = (comparison_df[model1] == comparison_df[model2]).mean()
                agreement_counts[f"{model1.split('_')[0]} vs {model2.split('_')[0]}"] = agreement
        
        # Create agreement dataframe
        agreement_df = pd.DataFrame({
            'Model Pair': list(agreement_counts.keys()),
            'Agreement': list(agreement_counts.values())
        })
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model Pair', y='Agreement', data=agreement_df)
        plt.title('Model Agreement (% of identical predictions)')
        plt.ylim(0, 1)
        for i, v in enumerate(agreement_df['Agreement']):
            plt.text(i, v + 0.01, f"{v:.2%}", ha='center')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('visualisations/comparison/model_agreement.png')
        
        # Save agreement data
        agreement_df.to_csv('compare/results/model_agreement.csv', index=False)
        
        # Calculate where all models agree vs. true label
        if len(models) >= 3:
            all_agree = (comparison_df[models[0]] == comparison_df[models[1]]) & (comparison_df[models[1]] == comparison_df[models[2]])
            comparison_df['All_Models_Agree'] = all_agree
            
            # Calculate accuracy when all models agree
            agree_df = comparison_df[all_agree]
            if len(agree_df) > 0:
                agree_accuracy = (agree_df[models[0]] == agree_df['True_Result']).mean()
                agree_pct = len(agree_df) / len(comparison_df)
                
                print(f"\nAll three models agree on {len(agree_df)} predictions ({agree_pct:.2%} of test set)")
                print(f"When all models agree, accuracy is {agree_accuracy:.4f}")
                
                # Create an ensemble prediction (majority vote)
                comparison_df['Ensemble_Pred'] = comparison_df[models].mode(axis=1)[0]
                ensemble_accuracy = accuracy_score(comparison_df['True_Result'], comparison_df['Ensemble_Pred'])
                print(f"\nEnsemble (majority vote) accuracy: {ensemble_accuracy:.4f}")
                
                # Add to results
                model_results['Model'].append('Ensemble')
                model_results['Accuracy'].append(ensemble_accuracy)
                
                f1_macro = f1_score(comparison_df['True_Result'], comparison_df['Ensemble_Pred'], average='macro')
                f1_weighted = f1_score(comparison_df['True_Result'], comparison_df['Ensemble_Pred'], average='weighted')
                model_results['F1_Score_Macro'].append(f1_macro)
                model_results['F1_Score_Weighted'].append(f1_weighted)
                
                # Get class-specific metrics
                report = classification_report(comparison_df['True_Result'], comparison_df['Ensemble_Pred'], output_dict=True)
                model_results['Home_Win_Precision'].append(report['H']['precision'])
                model_results['Home_Win_Recall'].append(report['H']['recall'])
                model_results['Draw_Precision'].append(report['D']['precision'])
                model_results['Draw_Recall'].append(report['D']['recall'])
                model_results['Away_Win_Precision'].append(report['A']['precision'])
                model_results['Away_Win_Recall'].append(report['A']['recall'])
                
                # Update results dataframe
                results_df = pd.DataFrame(model_results)
                results_df.to_csv(f'compare/results/model_comparison_with_ensemble.csv', index=False)
                
                # Update comparison visualization with ensemble
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Model', y='Accuracy', data=results_df)
                plt.title('Model Accuracy Comparison (with Ensemble)')
                plt.ylim(0, 1)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                for i, v in enumerate(results_df['Accuracy']):
                    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
                plt.tight_layout()
                plt.savefig(f'visualisations/comparison/accuracy_comparison_with_ensemble.png')

    print("\nComparison complete! Results saved to results/ directory.")

    # Additional visualizations
    
    # 1. Create a table visualization of model comparison results
    if os.path.exists('compare/results/model_comparison_with_ensemble.csv'):
        ensemble_results = pd.read_csv('compare/results/model_comparison_with_ensemble.csv')
        
        # Visualize as a color-coded table
        plt.figure(figsize=(14, 8))
        metrics_to_show = ['Accuracy', 'F1_Score_Weighted', 'Home_Win_Precision', 'Draw_Precision', 'Away_Win_Precision']
        table_data = ensemble_results[['Model'] + metrics_to_show].set_index('Model')
        
        # Generate the table plot
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False) 
        ax.yaxis.set_visible(False)
        
        # Create the table and add coloring
        table = plt.table(
            cellText=np.round(table_data.values, 4),
            rowLabels=table_data.index,
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.15] * len(table_data.columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Color code the cells based on values (higher is better)
        for i in range(len(table_data.index)):
            for j in range(len(table_data.columns)):
                cell = table[i+1, j]
                value = table_data.iloc[i, j]
                # Create a color gradient from white to green
                cell_color = (1 - value, 1, 1 - value)
                cell.set_facecolor(cell_color)
        
        plt.title('Model Comparison Summary', fontsize=16)
        plt.tight_layout()
        plt.savefig('visualisations/comparison/model_comparison_table.png', bbox_inches='tight')
    
    # 2. Visualize a sample of prediction comparison
    if os.path.exists('compare/results/prediction_comparison.csv'):
        predictions = pd.read_csv('compare/results/prediction_comparison.csv')
        
        # Show only the first 10 rows and the main prediction columns
        prediction_sample = predictions.tail(10)
        cols_to_show = ['True_Result'] + [col for col in predictions.columns if col.endswith('_Pred')]
        prediction_sample = prediction_sample[cols_to_show]
        
        # Create a visualization
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Create the table
        table = plt.table(
            cellText=prediction_sample.values,
            colLabels=prediction_sample.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.15] * len(prediction_sample.columns)
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Color code the cells based on match with true result
        for i in range(len(prediction_sample)):
            true_val = prediction_sample.iloc[i, 0]  # True_Result
            
            # Color the True_Result cell
            cell = table[i+1, 0]
            cell.set_facecolor((0.9, 0.9, 1))  # Light blue
            
            # Color the prediction cells
            for j in range(1, len(prediction_sample.columns)):
                cell = table[i+1, j]
                pred_val = prediction_sample.iloc[i, j]
                
                # Green if matches true value, light red if not
                if pred_val == true_val:
                    cell.set_facecolor((0.7, 1, 0.7))  # Light green
                else:
                    cell.set_facecolor((1, 0.7, 0.7))  # Light red
        
        plt.title('Sample of Predictions (Last 10 Matches)', fontsize=16)
        plt.tight_layout()
        plt.savefig('visualisations/comparison/prediction_sample.png', bbox_inches='tight')
        
        # 3. Create a visualization of prediction agreement by match
        if len(cols_to_show) > 2:  # If we have multiple prediction columns
            # Count how many models agree with the true result for each match
            agreement_counts = pd.DataFrame({'Match': range(1, len(predictions) + 1)})
            
            # Number of models that predicted correctly for each match
            correct_preds = (predictions[cols_to_show].iloc[:, 1:] == predictions['True_Result'].values[:, None]).sum(axis=1)
            agreement_counts['Correct_Models'] = correct_preds
            
            # Calculate percentage
            num_models = len(cols_to_show) - 1
            agreement_counts['Correct_Percentage'] = correct_preds / num_models * 100
            
            # Plot the percentage of models that predicted correctly for each match
            plt.figure(figsize=(15, 6))
            plt.bar(agreement_counts['Match'], agreement_counts['Correct_Percentage'], 
                   color=plt.cm.viridis(agreement_counts['Correct_Percentage']/100))
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axhline(y=100, color='k', linestyle='-', alpha=0.3)
            plt.xlabel('Match Number')
            plt.ylabel('Percentage of Models Correct (%)')
            plt.title('Model Agreement with True Result by Match')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('visualisations/comparison/match_agreement.png')
            
            # 4. Visualize prediction distribution for each model
            plt.figure(figsize=(12, 8))
            
            # Count prediction distribution for each model
            pred_dist = {}
            for col in cols_to_show[1:]:  # Skip True_Result
                model_name = col.split('_')[0]
                counts = predictions[col].value_counts().reindex(['H', 'D', 'A']).fillna(0)
                pred_dist[model_name] = counts
            
            # Create DataFrame for plotting
            pred_dist_df = pd.DataFrame(pred_dist)
            
            # Create a stacked bar chart
            pred_dist_df.plot(kind='bar', stacked=False, figsize=(12, 8))
            plt.title('Prediction Distribution by Model')
            plt.xlabel('Match Result')
            plt.ylabel('Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Model')
            plt.tight_layout()
            plt.savefig('visualisations/comparison/prediction_distribution.png')
            
            # 5. Visualize model confidence (probability) distribution
            # Get probability columns for each result type
            home_prob_cols = [col for col in predictions.columns if 'Home_Win_Prob' in col]
            draw_prob_cols = [col for col in predictions.columns if 'Draw_Prob' in col]
            away_prob_cols = [col for col in predictions.columns if 'Away_Win_Prob' in col]
            
            if home_prob_cols and draw_prob_cols and away_prob_cols:
                plt.figure(figsize=(15, 10))
                
                # Create subplots for each result type
                plt.subplot(3, 1, 1)
                predictions[home_prob_cols].plot(kind='kde', ax=plt.gca())
                plt.title('Home Win Probability Distribution')
                plt.xlabel('Probability')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.subplot(3, 1, 2)
                predictions[draw_prob_cols].plot(kind='kde', ax=plt.gca())
                plt.title('Draw Probability Distribution')
                plt.xlabel('Probability')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.subplot(3, 1, 3)
                predictions[away_prob_cols].plot(kind='kde', ax=plt.gca())
                plt.title('Away Win Probability Distribution')
                plt.xlabel('Probability')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig('visualisations/comparison/probability_distributions.png')

if __name__ == "__main__":
    main() 
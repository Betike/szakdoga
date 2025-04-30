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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler


def main():
    os.makedirs('compare/results', exist_ok=True)
    
    test_data = pd.read_csv("data/train_test/test_data_chronological.csv")
    print(f"Test data shape: {test_data.shape}")
    
    y_true = test_data['Result']

    pytorch_predictions = None
    xgboost_predictions = None
    randomforest_predictions = None

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

    def evaluate_model(model_type):
        print(f"\nEvaluating {model_type} model...")
        
        if model_type == 'PyTorch':
            try:

                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                from train.train_pytorch_model import MatchPredictionNN
                
                with open('utils/neural_network/feature_scaling_info.json', 'r') as f:
                    feature_info = json.load(f)
                
                feature_names = feature_info['feature_names']
                scaler = StandardScaler()
                scaler.mean_ = np.array(feature_info['scaler_mean'])
                scaler.scale_ = np.array(feature_info['scaler_scale'])
                scaler.feature_names_in_ = np.array(feature_names, dtype=object)
                scaler.n_features_in_ = len(feature_names)
                
                model_path = "models/pytorch_model.pth"
                if not os.path.exists(model_path):
                    print(f"PyTorch model not found at {model_path}. Skipping.")
                    return None
                
                input_size = len(feature_names)
                model = MatchPredictionNN(input_size)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                
                X_test = test_data[feature_names]
                X_test_scaled = scaler.transform(X_test)
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                
                with torch.no_grad():
                    outputs = model(X_test_tensor)
                    _, predicted = torch.max(outputs, 1)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                label_map_inverse = {0: 'A', 1: 'D', 2: 'H'}
                predictions = [label_map_inverse[p.item()] for p in predicted]
                
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
            try:
                model_path = "models/xgboost_prediction_model.json"
                if not os.path.exists(model_path):
                    print(f"XGBoost model not found at {model_path}. Skipping.")
                    return None
                    
                model = xgb.Booster()
                model.load_model(model_path)
                
                with open('utils/xgboost/xgboost_label_encoder.pkl', 'rb') as f:
                    label_encoder = pickle.load(f)
                
                feature_names = model.feature_names
                if feature_names is None:
                    feature_names = [col for col in test_data.columns 
                                    if (col.startswith('Home_') or col.startswith('Away_') or col.startswith('Diff_')) 
                                    and test_data[col].dtype in [np.int64, np.float64]
                                    and col != 'Home_CrdR' and col != 'Away_CrdR' and col != 'Diff_CrdR']
                
                X_test = test_data[feature_names]
                dtest = xgb.DMatrix(X_test)
                
                probabilities = model.predict(dtest)
                pred_indices = np.argmax(probabilities, axis=1)
                
                predictions = label_encoder.inverse_transform(pred_indices)
                
                label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
                
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
            try:
                model_path = "models/random_forest_prediction_model.pkl"
                if not os.path.exists(model_path):
                    print(f"Random Forest model not found at {model_path}. Skipping.")
                    return None
                    
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                with open('utils/random_forest/random_forest_label_encoder.pkl', 'rb') as f:
                    label_encoder = pickle.load(f)
                
                feature_path = "utils/random_forest/random_forest_features.json"
                if os.path.exists(feature_path):
                    with open(feature_path, 'r') as f:
                        feature_info = json.load(f)
                        feature_names = feature_info['feature_names']
                else:
                    try:
                        feature_names = model.feature_names_in_
                    except:
                        feature_names = [col for col in test_data.columns 
                                       if (col.startswith('Home_') or col.startswith('Away_') or col.startswith('Diff_')) 
                                       and test_data[col].dtype in [np.int64, np.float64]
                                       and col != 'Home_CrdR' and col != 'Away_CrdR' and col != 'Diff_CrdR']
                
                X_test = test_data[feature_names]
                
                probabilities = model.predict_proba(X_test)
                predictions = model.predict(X_test)
                
                predictions = label_encoder.inverse_transform(predictions)
                
                label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
                
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

    result_pytorch = evaluate_model('PyTorch')
    result_xgboost = evaluate_model('XGBoost')
    result_randomforest = evaluate_model('RandomForest')

    comparison_df = pd.DataFrame({'True_Result': y_true})

    if result_pytorch is not None:
        pytorch_predictions, pytorch_probs = result_pytorch
        comparison_df['PyTorch_Pred'] = pytorch_predictions
        comparison_df = pd.concat([comparison_df, pytorch_probs.add_prefix('PyTorch_')], axis=1)
        
        accuracy = accuracy_score(y_true, pytorch_predictions)
        f1_macro = f1_score(y_true, pytorch_predictions, average='macro')
        f1_weighted = f1_score(y_true, pytorch_predictions, average='weighted')
        
        report = classification_report(y_true, pytorch_predictions, output_dict=True)
        
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
        
        accuracy = accuracy_score(y_true, xgboost_predictions)
        f1_macro = f1_score(y_true, xgboost_predictions, average='macro')
        f1_weighted = f1_score(y_true, xgboost_predictions, average='weighted')
        
        report = classification_report(y_true, xgboost_predictions, output_dict=True)
        
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
        
        accuracy = accuracy_score(y_true, randomforest_predictions)
        f1_macro = f1_score(y_true, randomforest_predictions, average='macro')
        f1_weighted = f1_score(y_true, randomforest_predictions, average='weighted')
        
        report = classification_report(y_true, randomforest_predictions, output_dict=True)
        
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

    results_df = pd.DataFrame(model_results)
    print("\nSummary:")
    print(results_df[['Model', 'Accuracy', 'F1_Score_Macro', 'F1_Score_Weighted']])

    results_df.to_csv(f'compare/results/model_comparison.csv', index=False)
    comparison_df.to_csv(f'compare/results/prediction_comparison.csv', index=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='Accuracy', data=results_df)
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(results_df['Accuracy']):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig(f'visualisations/comparison/accuracy_comparison.png')

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

    plt.figure(figsize=(15, 10))

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

    if len(model_results['Model']) > 1:
        agreement_counts = {}
        
        models = [m for m in ['PyTorch_Pred', 'XGBoost_Pred', 'RandomForest_Pred'] if m in comparison_df.columns]
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                agreement = (comparison_df[model1] == comparison_df[model2]).mean()
                agreement_counts[f"{model1.split('_')[0]} vs {model2.split('_')[0]}"] = agreement
        
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
        
        agreement_df.to_csv('compare/results/model_agreement.csv', index=False)
        
        if len(models) >= 3:
            all_agree = (comparison_df[models[0]] == comparison_df[models[1]]) & (comparison_df[models[1]] == comparison_df[models[2]])
            comparison_df['All_Models_Agree'] = all_agree
            
            agree_df = comparison_df[all_agree]
            if len(agree_df) > 0:
                agree_accuracy = (agree_df[models[0]] == agree_df['True_Result']).mean()
                agree_pct = len(agree_df) / len(comparison_df)
                
                print(f"\nAll models agree on {len(agree_df)} predictions ({agree_pct:.2%} of test set)")
                print(f"When all models agree, accuracy is {agree_accuracy:.4f}")
                
                comparison_df['Ensemble_Pred'] = comparison_df[models].mode(axis=1)[0]
                ensemble_accuracy = accuracy_score(comparison_df['True_Result'], comparison_df['Ensemble_Pred'])
                print(f"\nEnsemble accuracy: {ensemble_accuracy:.4f}")
                
                model_results['Model'].append('Ensemble')
                model_results['Accuracy'].append(ensemble_accuracy)
                
                f1_macro = f1_score(comparison_df['True_Result'], comparison_df['Ensemble_Pred'], average='macro')
                f1_weighted = f1_score(comparison_df['True_Result'], comparison_df['Ensemble_Pred'], average='weighted')
                model_results['F1_Score_Macro'].append(f1_macro)
                model_results['F1_Score_Weighted'].append(f1_weighted)
                
                report = classification_report(comparison_df['True_Result'], comparison_df['Ensemble_Pred'], output_dict=True)
                model_results['Home_Win_Precision'].append(report['H']['precision'])
                model_results['Home_Win_Recall'].append(report['H']['recall'])
                model_results['Draw_Precision'].append(report['D']['precision'])
                model_results['Draw_Recall'].append(report['D']['recall'])
                model_results['Away_Win_Precision'].append(report['A']['precision'])
                model_results['Away_Win_Recall'].append(report['A']['recall'])
                
                results_df = pd.DataFrame(model_results)
                results_df.to_csv(f'compare/results/model_comparison_with_ensemble.csv', index=False)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Model', y='Accuracy', data=results_df)
                plt.title('Model Accuracy Comparison (with Ensemble)')
                plt.ylim(0, 1)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                for i, v in enumerate(results_df['Accuracy']):
                    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
                plt.tight_layout()
                plt.savefig(f'visualisations/comparison/accuracy_comparison_with_ensemble.png')

    print("\nComparison complete")
    
    if os.path.exists('compare/results/model_comparison_with_ensemble.csv'):
        ensemble_results = pd.read_csv('compare/results/model_comparison_with_ensemble.csv')
        
        plt.figure(figsize=(14, 8))
        metrics_to_show = ['Accuracy', 'F1_Score_Weighted', 'Home_Win_Precision', 'Draw_Precision', 'Away_Win_Precision']
        table_data = ensemble_results[['Model'] + metrics_to_show].set_index('Model')
        
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False) 
        ax.yaxis.set_visible(False)
        
        table = plt.table(
            cellText=np.round(table_data.values, 4),
            rowLabels=table_data.index,
            colLabels=table_data.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.15] * len(table_data.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        for i in range(len(table_data.index)):
            for j in range(len(table_data.columns)):
                cell = table[i+1, j]
                value = table_data.iloc[i, j]
                cell_color = (1 - value, 1, 1 - value)
                cell.set_facecolor(cell_color)
        
        plt.title('Model Comparison Summary', fontsize=16)
        plt.tight_layout()
        plt.savefig('visualisations/comparison/model_comparison_table.png', bbox_inches='tight')
    
    if os.path.exists('compare/results/prediction_comparison.csv'):
        predictions = pd.read_csv('compare/results/prediction_comparison.csv')
        
        prediction_sample = predictions.tail(10)
        cols_to_show = ['True_Result'] + [col for col in predictions.columns if col.endswith('_Pred')]
        prediction_sample = prediction_sample[cols_to_show]
        
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        table = plt.table(
            cellText=prediction_sample.values,
            colLabels=prediction_sample.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.15] * len(prediction_sample.columns)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        for i in range(len(prediction_sample)):
            true_val = prediction_sample.iloc[i, 0]
            
            cell = table[i+1, 0]
            cell.set_facecolor((0.9, 0.9, 1))
            
            for j in range(1, len(prediction_sample.columns)):
                cell = table[i+1, j]
                pred_val = prediction_sample.iloc[i, j]
                
                if pred_val == true_val:
                    cell.set_facecolor((0.7, 1, 0.7))
                else:
                    cell.set_facecolor((1, 0.7, 0.7))  # Light red
        
        plt.title('Sample of Predictions (Last 10 Matches)', fontsize=16)
        plt.tight_layout()
        plt.savefig('visualisations/comparison/prediction_sample.png', bbox_inches='tight')
        
        if len(cols_to_show) > 2:
            agreement_counts = pd.DataFrame({'Match': range(1, len(predictions) + 1)})
            
            correct_preds = (predictions[cols_to_show].iloc[:, 1:] == predictions['True_Result'].values[:, None]).sum(axis=1)
            agreement_counts['Correct_Models'] = correct_preds
            
            num_models = len(cols_to_show) - 1
            agreement_counts['Correct_Percentage'] = correct_preds / num_models * 100
            
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
            
            plt.figure(figsize=(12, 8))
            
            pred_dist = {}
            for col in cols_to_show[1:]:
                model_name = col.split('_')[0]
                counts = predictions[col].value_counts().reindex(['H', 'D', 'A']).fillna(0)
                pred_dist[model_name] = counts
            
            pred_dist_df = pd.DataFrame(pred_dist)
            
            pred_dist_df.plot(kind='bar', stacked=False, figsize=(12, 8))
            plt.title('Prediction Distribution by Model')
            plt.xlabel('Match Result')
            plt.ylabel('Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Model')
            plt.tight_layout()
            plt.savefig('visualisations/comparison/prediction_distribution.png')
            
            home_prob_cols = [col for col in predictions.columns if 'Home_Win_Prob' in col]
            draw_prob_cols = [col for col in predictions.columns if 'Draw_Prob' in col]
            away_prob_cols = [col for col in predictions.columns if 'Away_Win_Prob' in col]
            
            if home_prob_cols and draw_prob_cols and away_prob_cols:
                plt.figure(figsize=(15, 10))
                
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
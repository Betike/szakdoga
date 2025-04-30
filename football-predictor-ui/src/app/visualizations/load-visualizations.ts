import { readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

export interface ModelStrengths {
  [key: string]: string[];
}

export interface ModelColors {
  [key: string]: string;
}

export interface VisualizationPath {
  exists: boolean;
  path: string;
}

export interface VisualizationPaths {
  [key: string]: VisualizationPath;
}

export interface ModelEntry {
  name: string;
  accuracy: number;
  f1_macro: number;
  f1_weighted: number;
  home_win_precision: number;
  home_win_recall: number;
  draw_precision: number;
  draw_recall: number;
  away_win_precision: number;
  away_win_recall: number;
  strengths?: string[];
  color?: string;
}

export function getModelColors(): ModelColors {
  return {
    'XGBoost': "bg-blue-500",
    'RandomForest': "bg-green-500", 
    'PyTorch': "bg-purple-500",
    'Ensemble': "bg-yellow-500"
  };
}

export function getModelStrengths(): ModelStrengths {
  return {
    'XGBoost': ["Speed"],
    'RandomForest': ["Home and away win prediction"],
    'PyTorch': ["Draw prediction"],
    'Ensemble': ["Best overall accuracy"]
  };
}

export async function getVisualizationPaths(): Promise<VisualizationPaths> {
  try {
    const visualizationMap: Record<string, string> = {
      accuracy: 'accuracy_comparison_with_ensemble.png',
      f1: 'f1_comparison.png',
      confusion: 'class_metrics_comparison.png',
      probabilities: 'probability_distributions.png',
      prediction: 'prediction_distribution.png',
      agreement: 'model_agreement.png',
      sample: 'prediction_sample.png',
      table: 'model_comparison_table.png',
      match: 'match_agreement.png',
      
      nn_confusion: 'neural_network/confusion_matrix.png',
      nn_training: 'neural_network/training_history.png',
      
      rf_confusion: 'random_forest/confusion_matrix.png',
      rf_importance: 'random_forest/feature_importance.png',
      
      xgb_confusion: 'xgboost/confusion_matrix.png',
      xgb_importance: 'xgboost/feature_importance.png'
    };
  
    const paths: VisualizationPaths = {};
    const baseDir = join(process.cwd(), 'public', 'images', 'visualizations');
  
    try {
      if (!existsSync(baseDir)) {
        mkdirSync(baseDir, { recursive: true });
      }
    } catch (e) {
      console.log("Unable to create directory, likely running in browser context. Error: ", e);
    }
    
    for (const [key, filename] of Object.entries(visualizationMap)) {
      let fileExists = false;
      try {
        const filePath = join(baseDir, filename);
        fileExists = existsSync(filePath);
      } catch (e) {
        fileExists = true;
        console.log("Unable to check file existence, likely running in browser context. Error: ", e);
      }
      
      const publicPath = `/images/visualizations/${filename}`;
      
      paths[key] = {
        exists: fileExists,
        path: publicPath
      };
    }
    
    return paths;
  } catch (error) {
    console.error("Error getting visualization paths:", error);
    return {};
  }
}

export async function loadModelPerformanceData(): Promise<ModelEntry[] | null> {
  try {
    const possiblePaths = [
      join(process.cwd(), 'public', 'data', 'model_comparison_with_ensemble.csv'),
      join(process.cwd(), '..', 'compare', 'results', 'model_comparison_with_ensemble.csv'),
      join(process.cwd(), 'compare', 'results', 'model_comparison_with_ensemble.csv'),
      join(process.cwd(), 'results', 'model_comparison_with_ensemble.csv')
    ];
    
    let csvData: string | null = null;
    let foundPath: string | null = null;
    
    console.log("Searching for model comparison CSV in the following locations:");
    possiblePaths.forEach(path => console.log(`- ${path}`));
    
    for (const csvPath of possiblePaths) {
      try {
        console.log(`Checking path: ${csvPath}`);
        if (existsSync(csvPath)) {
          foundPath = csvPath;
          csvData = readFileSync(csvPath, 'utf-8');
          console.log(`✓ Found model comparison CSV at: ${csvPath}`);
          break;
        } else {
          console.log(`✗ File not found at: ${csvPath}`);
        }
      } catch (e) {
        console.warn(`Error checking path ${csvPath}:`, e);
      }
    }
    
    if (!csvData || !foundPath) {
      console.error('Model comparison CSV not found in any of the checked locations');
      return null;
    }
    
    const lines = csvData.split('\n').filter(Boolean);
    const headers = lines[0].split(',');
    
    const modelData: ModelEntry[] = lines.slice(1).map(line => {
      const values = line.split(',');
      const modelEntry: Record<string, string> = {};
      
      headers.forEach((header, index) => {
        if (index < values.length) {
          modelEntry[header.trim()] = values[index]?.trim() || '';
        }
      });
      
      return {
        name: modelEntry['Model'] || '',
        accuracy: parseFloat(modelEntry['Accuracy'] || '0') * 100,
        f1_macro: parseFloat(modelEntry['F1_Score_Macro'] || '0') * 100,
        f1_weighted: parseFloat(modelEntry['F1_Score_Weighted'] || '0') * 100,
        home_win_precision: parseFloat(modelEntry['Home_Win_Precision'] || '0') * 100,
        home_win_recall: parseFloat(modelEntry['Home_Win_Recall'] || '0') * 100,
        draw_precision: parseFloat(modelEntry['Draw_Precision'] || '0') * 100,
        draw_recall: parseFloat(modelEntry['Draw_Recall'] || '0') * 100,
        away_win_precision: parseFloat(modelEntry['Away_Win_Precision'] || '0') * 100,
        away_win_recall: parseFloat(modelEntry['Away_Win_Recall'] || '0') * 100,
        color: getColorForModel(modelEntry['Model'] || ''),
        strengths: getStrengthsForModel(modelEntry['Model'] || '')
      };
    });
    
    return modelData;
  } catch (error) {
    console.error('Error loading model performance data:', error);
    return null;
  }
}

function getColorForModel(modelName: string): string {
  const colors = getModelColors();
  const nameMapping: Record<string, string> = {
    'PyTorch': 'PyTorch',
    'XGBoost': 'XGBoost',
    'RandomForest': 'RandomForest',
    'Ensemble': 'Ensemble',
    'Random Forest': 'RandomForest',
    'Neural Network': 'PyTorch'
  };
  
  const mappedName = nameMapping[modelName] || modelName;
  return colors[mappedName] || colors['PyTorch'] || '';
}

function getStrengthsForModel(modelName: string): string[] {
  const strengths = getModelStrengths();
  const nameMapping: Record<string, string> = {
    'PyTorch': 'PyTorch',
    'XGBoost': 'XGBoost',
    'RandomForest': 'RandomForest',
    'Ensemble': 'Ensemble',
    'Random Forest': 'RandomForest',
    'Neural Network': 'PyTorch'
  };
  
  const mappedName = nameMapping[modelName] || modelName;
  return strengths[mappedName] || strengths['PyTorch'] || [];
} 
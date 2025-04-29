// Server-only functions for data loading with browser-safe fallbacks
import { readFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

// TypeScript interfaces
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

/**
 * Get model colors for visualization
 */
export function getModelColors(): ModelColors {
  return {
    'XGBoost': "bg-blue-500",
    'RandomForest': "bg-green-500", 
    'PyTorch': "bg-purple-500",
    'Ensemble': "bg-yellow-500"
  };
}

/**
 * Default model strengths if data isn't available
 */
export function getModelStrengths(): ModelStrengths {
  return {
    'XGBoost': ["Speed"],
    'RandomForest': ["Home and away win prediction"],
    'PyTorch': ["Draw prediction"],
    'Ensemble': ["Best overall accuracy"]
  };
}

/**
 * Get paths to visualization images
 */
export async function getVisualizationPaths(): Promise<VisualizationPaths> {
  try {
    // Define static visualization paths and check existence on the server
    const visualizationMap: Record<string, string> = {
      // Comparison visualizations
      accuracy: 'accuracy_comparison_with_ensemble.png',
      f1: 'f1_comparison.png',
      confusion: 'class_metrics_comparison.png',
      probabilities: 'probability_distributions.png',
      prediction: 'prediction_distribution.png',
      agreement: 'model_agreement.png',
      sample: 'prediction_sample.png',
      table: 'model_comparison_table.png',
      match: 'match_agreement.png',
      
      // Neural Network visualizations
      nn_confusion: 'neural_network/confusion_matrix.png',
      nn_training: 'neural_network/training_history.png',
      
      // Random Forest visualizations
      rf_confusion: 'random_forest/confusion_matrix.png',
      rf_importance: 'random_forest/feature_importance.png',
      
      // XGBoost visualizations
      xgb_confusion: 'xgboost/confusion_matrix.png',
      xgb_importance: 'xgboost/feature_importance.png'
    };
  
    // Create paths object
    const paths: VisualizationPaths = {};
    const baseDir = join(process.cwd(), 'public', 'images', 'visualizations');
  
    // If directory doesn't exist yet, create it
    try {
      if (!existsSync(baseDir)) {
        mkdirSync(baseDir, { recursive: true });
      }
    } catch (e) {
      console.log("Unable to create directory, likely running in browser context. Error: ", e);
    }
    
    // Check each visualization file
    for (const [key, filename] of Object.entries(visualizationMap)) {
      // For client-side safety, avoid using existsSync which is a server-only function
      let fileExists = false;
      try {
        const filePath = join(baseDir, filename);
        fileExists = existsSync(filePath);
      } catch (e) {
        // In client context, we fall back to assuming files exist
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
    // Return empty paths in case of error
    return {};
  }
}

/**
 * Load model performance data from the comparison CSV
 */
export async function loadModelPerformanceData(): Promise<ModelEntry[] | null> {
  try {
    // Check multiple possible paths for the CSV file
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
    
    // Try each path until we find the file
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
        // Continue to next path
      }
    }
    
    // If file not found in any location
    if (!csvData || !foundPath) {
      console.error('Model comparison CSV not found in any of the checked locations');
      return null;
    }
    
    // Parse CSV (simple parser)
    const lines = csvData.split('\n').filter(Boolean);
    const headers = lines[0].split(',');
    
    // Extract model data
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

/**
 * Helper function to get color for a model name
 */
function getColorForModel(modelName: string): string {
  const colors = getModelColors();
  // Map model names from CSV to our color keys
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

/**
 * Helper function to get strengths for a model name
 */
function getStrengthsForModel(modelName: string): string[] {
  const strengths = getModelStrengths();
  // Map model names from CSV to our strength keys
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
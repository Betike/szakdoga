const fs = require('fs');
const path = require('path');

// Paths
const COMPARE_RESULTS_DIR = path.join(__dirname, 'compare', 'results');
const COMPARE_VISUAL_DIR = path.join(__dirname, 'visualisations', 'comparison');
const TARGET_DIR = path.join(__dirname, 'football-predictor-ui', 'public', 'images', 'visualizations');
const TARGET_DATA_DIR = path.join(__dirname, 'football-predictor-ui', 'public', 'data');

// Model-specific visualization paths
const NEURAL_NET_DIR = path.join(__dirname, 'visualisations', 'neural_network');
const RANDOM_FOREST_DIR = path.join(__dirname, 'visualisations', 'random_forest');
const XGBOOST_DIR = path.join(__dirname, 'visualisations', 'xgboost');

// Target directories for model-specific visualizations
const TARGET_NEURAL_NET_DIR = path.join(TARGET_DIR, 'neural_network');
const TARGET_RANDOM_FOREST_DIR = path.join(TARGET_DIR, 'random_forest');
const TARGET_XGBOOST_DIR = path.join(TARGET_DIR, 'xgboost');

// Create target directories if they don't exist
fs.mkdirSync(TARGET_DIR, { recursive: true });
fs.mkdirSync(TARGET_DATA_DIR, { recursive: true });
fs.mkdirSync(TARGET_NEURAL_NET_DIR, { recursive: true });
fs.mkdirSync(TARGET_RANDOM_FOREST_DIR, { recursive: true });
fs.mkdirSync(TARGET_XGBOOST_DIR, { recursive: true });

// File mappings - source to destination
const visualizationFiles = [
  // From compare/results
  { src: path.join(COMPARE_VISUAL_DIR, 'accuracy_comparison_with_ensemble.png'), dest: path.join(TARGET_DIR, 'accuracy_comparison_with_ensemble.png') },
  { src: path.join(COMPARE_VISUAL_DIR, 'f1_comparison.png'), dest: path.join(TARGET_DIR, 'f1_comparison.png') },
  { src: path.join(COMPARE_VISUAL_DIR, 'class_metrics_comparison.png'), dest: path.join(TARGET_DIR, 'class_metrics_comparison.png') },
  { src: path.join(COMPARE_VISUAL_DIR, 'model_agreement.png'), dest: path.join(TARGET_DIR, 'model_agreement.png') },
  { src: path.join(COMPARE_VISUAL_DIR, 'prediction_sample.png'), dest: path.join(TARGET_DIR, 'prediction_sample.png') },
  { src: path.join(COMPARE_VISUAL_DIR, 'model_comparison_table.png'), dest: path.join(TARGET_DIR, 'model_comparison_table.png') },
  { src: path.join(COMPARE_VISUAL_DIR, 'match_agreement.png'), dest: path.join(TARGET_DIR, 'match_agreement.png') },
  { src: path.join(COMPARE_RESULTS_DIR, 'model_comparison_with_ensemble.csv'), dest: path.join(TARGET_DATA_DIR, 'model_comparison_with_ensemble.csv') },
  { src: path.join(COMPARE_RESULTS_DIR, 'prediction_comparison.csv'), dest: path.join(TARGET_DATA_DIR, 'prediction_comparison.csv') },
  
  // From visualisations/comparison
  { src: path.join(COMPARE_VISUAL_DIR, 'prediction_distribution.png'), dest: path.join(TARGET_DIR, 'prediction_distribution.png') },
  { src: path.join(COMPARE_VISUAL_DIR, 'probability_distributions.png'), dest: path.join(TARGET_DIR, 'probability_distributions.png') },
  
  // From visualisations/neural_network
  { src: path.join(NEURAL_NET_DIR, 'confusion_matrix.png'), dest: path.join(TARGET_NEURAL_NET_DIR, 'confusion_matrix.png') },
  { src: path.join(NEURAL_NET_DIR, 'training_history.png'), dest: path.join(TARGET_NEURAL_NET_DIR, 'training_history.png') },
  
  // From visualisations/random_forest
  { src: path.join(RANDOM_FOREST_DIR, 'random_forest_confusion_matrix.png'), dest: path.join(TARGET_RANDOM_FOREST_DIR, 'confusion_matrix.png') },
  { src: path.join(RANDOM_FOREST_DIR, 'random_forest_feature_importance.png'), dest: path.join(TARGET_RANDOM_FOREST_DIR, 'feature_importance.png') },
  
  // From visualisations/xgboost
  { src: path.join(XGBOOST_DIR, 'xgboost_confusion_matrix.png'), dest: path.join(TARGET_XGBOOST_DIR, 'confusion_matrix.png') },
  { src: path.join(XGBOOST_DIR, 'xgboost_feature_importance.png'), dest: path.join(TARGET_XGBOOST_DIR, 'feature_importance.png') },
];

// Copy each file
console.log('Copying visualization files:');
let successCount = 0;
let errorCount = 0;

visualizationFiles.forEach(file => {
  try {
    if (fs.existsSync(file.src)) {
      fs.copyFileSync(file.src, file.dest);
      console.log(`✓ Copied: ${path.basename(file.src)}`);
      successCount++;
    } else {
      console.log(`× Missing: ${path.basename(file.src)}`);
      errorCount++;
    }
  } catch (error) {
    console.error(`Error copying ${file.src}: ${error.message}`);
    errorCount++;
  }
});

console.log(`\nComplete! ${successCount} files copied, ${errorCount} files failed.`);
console.log(`Files were copied to: ${TARGET_DIR} and ${TARGET_DATA_DIR}`); 
// Simple script to test the prediction API
const fs = require('fs');
const { exec } = require('child_process');
const path = require('path');

// Get parent directory
const rootDir = process.cwd();
const parentDir = path.join(rootDir, '..');
const predictScriptPath = path.join(parentDir, 'predict', 'predict_with_xgboost.py');

console.log('Current working directory:', rootDir);
console.log('Parent directory:', parentDir);
console.log('Script path:', predictScriptPath);
console.log('Script exists:', fs.existsSync(predictScriptPath));

// Execute the Python script from the parent directory
const cmd = `cd "${parentDir}" && python "predict/predict_with_xgboost.py" --single-match --home "Arsenal" --away "Liverpool" --json`;
console.log('Executing command:', cmd);

exec(cmd, (error, stdout, stderr) => {
  if (error) {
    console.error('Error executing Python script:', error);
    return;
  }
  
  if (stderr) {
    console.error('Script stderr:', stderr);
  }
  
  console.log('Script output:');
  console.log(stdout);
  
  try {
    const result = JSON.parse(stdout);
    console.log('Parsed result:', result);
  } catch (e) {
    console.error('Error parsing JSON:', e);
  }
}); 
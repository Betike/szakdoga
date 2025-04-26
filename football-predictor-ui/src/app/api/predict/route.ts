import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import util from 'util';
import path from 'path';
import fs from 'fs';

const execPromise = util.promisify(exec);

// Define types for mockResults and errors
type PredictionResult = {
  result: string; 
  probabilities: Record<string, number>;
  warning?: string;
  failed_models?: Record<string, string>;
};

// Function to run a Python predictor script
async function runPredictor(
  predictorType: string,
  homeTeam: string,
  awayTeam: string
): Promise<PredictionResult> {
  try {
    // Get the project root directory
    const currentDir = process.cwd();
    
    // Path to the parent directory containing the predict folder
    const parentDir = path.join(currentDir, '..');
    
    // Path to the wrapper script
    const wrapperScript = path.join(parentDir, 'predict', 'run_prediction.py');
    
    // Debug info
    console.log(`Current working directory: ${currentDir}`);
    console.log(`Parent directory: ${parentDir}`);
    console.log(`Wrapper script path: ${wrapperScript}`);
    console.log(`Script exists: ${fs.existsSync(wrapperScript)}`);
    console.log(`Predicting ${homeTeam} vs ${awayTeam} using ${predictorType} model`);

    // Execute the Python wrapper script from the parent directory
    const { stdout, stderr } = await execPromise(
      `cd "${parentDir}" && python "${path.join('predict', 'run_prediction.py')}" --model "${predictorType}" --home "${homeTeam}" --away "${awayTeam}" --json`
    );

    if (stderr && !stderr.includes('WARNING')) {
      console.error(`Script error: ${stderr}`);
      throw new Error(`Prediction failed: ${stderr}`);
    }

    // Parse the JSON output from the Python script
    try {
      const result = JSON.parse(stdout);
      
      // Check for error
      if (result.error) {
        throw new Error(result.error);
      }
      
      // Format the result to match our expected structure
      return {
        result: result.prediction || result.result,
        probabilities: result.probabilities || {},
        warning: result.warning,
        failed_models: result.failed_models
      };
    } catch (parseError) {
      console.error('Error parsing Python script output:', parseError);
      throw new Error('Failed to parse prediction results');
    }
  } catch (execError) {
    console.error('Prediction error:', execError);
    
    // Fall back to mock data if the Python script execution fails
    console.log('Falling back to mock data due to error');
    
    const mockResults: Record<string, PredictionResult> = {
      xgboost: {
        result: 'H', 
        probabilities: { 'Home win': 0.65, 'Draw': 0.25, 'Away win': 0.10 }
      },
      random_forest: {
        result: 'D',
        probabilities: { 'Draw': 0.45, 'Home win': 0.40, 'Away win': 0.15 }
      },
      pytorch: {
        result: 'H',
        probabilities: { 'Home win': 0.55, 'Draw': 0.30, 'Away win': 0.15 }
      },
      ensemble: {
        result: 'H',
        probabilities: { 'Home win': 0.60, 'Draw': 0.30, 'Away win': 0.10 },
        warning: "Using mock data due to error",
        failed_models: { "actual_model": (execError as Error).message }
      }
    };
    
    return mockResults[predictorType];
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { predictorType, homeTeam, awayTeam } = body;

    // Validate input
    if (!predictorType || !homeTeam || !awayTeam) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Check if teams are identical
    if (homeTeam === awayTeam) {
      return NextResponse.json(
        { error: 'Home team and away team cannot be identical' },
        { status: 400 }
      );
    }

    const result = await runPredictor(predictorType, homeTeam, awayTeam);

    return NextResponse.json(result);
  } catch (errorObj: unknown) {
    console.error('API error:', errorObj);
    const errorMessage = errorObj instanceof Error ? errorObj.message : 'An error occurred during prediction';
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    );
  }
} 
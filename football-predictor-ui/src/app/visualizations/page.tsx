import { 
  getModelColors, 
  getModelStrengths, 
  getVisualizationPaths,
  loadModelPerformanceData,
  ModelEntry,
  VisualizationPaths
} from "./load-visualizations";
import { VisualizationTabs } from "@/components/VisualizationTabs";

export default async function VisualizationsPage() {
  const visualizationPaths: VisualizationPaths = await getVisualizationPaths();
  const modelStrengths = getModelStrengths();
  const modelColors = getModelColors();
  
  let modelPerformance: ModelEntry[] = await loadModelPerformanceData() || [];
  
  if (modelPerformance.length === 0) {
    console.warn("Falling back to hardcoded model performance data");
    modelPerformance = [
      {
        name: "XGBoost",
        accuracy: 58.17,
        f1_macro: 49.50,
        f1_weighted: 54.44,
        strengths: modelStrengths["XGBoost"],
        color: modelColors["XGBoost"],
        home_win_precision: 58.0,
        home_win_recall: 64.0,
        draw_precision: 41.0,
        draw_recall: 13.0,
        away_win_precision: 60.0,
        away_win_recall: 78.0
      },
      {
        name: "Random Forest",
        accuracy: 60.56,
        f1_macro: 54.54,
        f1_weighted: 58.58,
        strengths: modelStrengths["RandomForest"],
        color: modelColors["RandomForest"],
        home_win_precision: 60.0,
        home_win_recall: 68.0,
        draw_precision: 43.0,
        draw_recall: 23.0,
        away_win_precision: 66.0,
        away_win_recall: 75.0
      },
      {
        name: "Neural Network",
        accuracy: 60.00,
        f1_macro: 48.92,
        f1_weighted: 53.18,
        strengths: modelStrengths["PyTorch"],
        color: modelColors["PyTorch"],
        home_win_precision: 57.0,
        home_win_recall: 62.0,
        draw_precision: 40.0,
        draw_recall: 14.0,
        away_win_precision: 58.0,
        away_win_recall: 76.0
      },
      {
        name: "Ensemble",
        accuracy: 62.11,
        f1_macro: 55.78,
        f1_weighted: 59.94,
        strengths: modelStrengths["Ensemble"],
        color: modelColors["Ensemble"],
        home_win_precision: 62.0,
        home_win_recall: 70.0,
        draw_precision: 45.0,
        draw_recall: 24.0,
        away_win_precision: 68.0,
        away_win_recall: 76.0
      }
    ];
  }

  return (
    <main className="flex flex-col items-center p-6 md:p-10">
      <div className="max-w-5xl w-full">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-3">
            Model Visualizations
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Explore performance metrics and visualizations of our prediction models.
          </p>
        </div>

        <VisualizationTabs 
          modelPerformance={modelPerformance} 
          visualizationPaths={visualizationPaths} 
        />
      </div>
    </main>
  );
} 
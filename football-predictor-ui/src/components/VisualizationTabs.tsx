"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Image from "next/image";
import { ModelEntry, VisualizationPaths } from "@/app/visualizations/load-visualizations";

interface VisualizationTabsProps {
  modelPerformance: ModelEntry[];
  visualizationPaths: VisualizationPaths;
}

export function VisualizationTabs({ modelPerformance, visualizationPaths }: VisualizationTabsProps) {
  return (
    <Tabs defaultValue="performance" className="w-full">
      <TabsList className="grid w-full grid-cols-4">
        <TabsTrigger value="performance">Model Performance</TabsTrigger>
        <TabsTrigger value="comparison">Model Comparison</TabsTrigger>
        <TabsTrigger value="other">Other Visualizations</TabsTrigger>
        <TabsTrigger value="model-details">Model Details</TabsTrigger>
      </TabsList>
      
      <TabsContent value="performance" className="mt-6">
        <h2 className="text-2xl font-semibold mb-4">Model Performance Metrics</h2>
        
        <div className="grid md:grid-cols-2 gap-6">
          {modelPerformance.map((model) => (
            <Card key={model.name}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center">
                    <div className={`w-3 h-3 rounded-full ${model.color} mr-2`}></div>
                    {model.name}
                  </CardTitle>
                  <Badge variant="outline">{model.accuracy.toFixed(2)}% Accuracy</Badge>
                </div>
                <CardDescription>F1 Score (Macro): {model.f1_macro.toFixed(2)}%</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div>
                    <p className="text-sm font-medium">Strength:</p>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {model.strengths?.map((strength) => (
                        <Badge key={strength} variant="secondary">{strength}</Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        
        <div className="mt-8 border rounded-lg overflow-hidden">
          <div className="p-4 bg-gray-50 border-b">
            <h3 className="font-semibold">Accuracy Comparison</h3>
          </div>
          <div className="p-6 flex justify-center">
            {visualizationPaths.accuracy?.exists ? (
              <div className="relative w-full aspect-video">
                <Image 
                  src={visualizationPaths.accuracy.path}
                  alt="Accuracy comparison chart" 
                  fill
                  style={{ objectFit: 'contain' }}
                />
              </div>
            ) : (
              <div className="w-full h-64 bg-gray-100 flex items-center justify-center">
                <p className="text-gray-400">Accuracy comparison visualization not available</p>
              </div>
            )}
          </div>
        </div>
      </TabsContent>

      <TabsContent value="comparison" className="mt-6">
        <h2 className="text-2xl font-semibold mb-4">Model Comparison Visualizations</h2>
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>F1 Score Comparison</CardTitle>
              <CardDescription>Macro and weighted F1 scores across all models</CardDescription>
            </CardHeader>
            <CardContent>
              {visualizationPaths.f1?.exists ? (
                <div className="relative w-full aspect-video">
                  <Image 
                    src={visualizationPaths.f1.path}
                    alt="F1 score comparison chart" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              ) : (
                <div className="w-full h-64 bg-gray-100 flex items-center justify-center">
                  <p className="text-gray-400">F1 score visualization not available</p>
                </div>
              )}
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Class-Specific Metrics</CardTitle>
              <CardDescription>Precision and recall for the predictions</CardDescription>
            </CardHeader>
            <CardContent>
              {visualizationPaths.confusion?.exists ? (
                <div className="relative w-full aspect-video">
                  <Image 
                    src={visualizationPaths.confusion.path}
                    alt="Class metrics chart" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              ) : (
                <div className="w-full h-64 bg-gray-100 flex items-center justify-center">
                  <p className="text-gray-400">Class metrics visualization not available</p>
                </div>
              )}
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Prediction Distribution</CardTitle>
              <CardDescription>Distribution of predicted match outcomes by model</CardDescription>
            </CardHeader>
            <CardContent>
              {visualizationPaths.prediction?.exists ? (
                <div className="relative w-full aspect-video">
                  <Image 
                    src={visualizationPaths.prediction.path}
                    alt="Prediction distribution chart" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              ) : (
                <div className="w-full h-64 bg-gray-100 flex items-center justify-center">
                  <p className="text-gray-400">Prediction distribution visualization not available</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        <div className="mt-8 p-4 border rounded-lg bg-gray-50">
          <h3 className="text-lg font-semibold mb-2">About These Visualizations</h3>
          <p className="text-gray-600">
            These visualizations are generated from the model comparison analysis on a test set of Premier League matches. 
            They help illustrate the relative strengths and weaknesses of each prediction model and highlight where they 
            agree or disagree on specific match outcomes. These metrics can differ based on the data used to train the models and <span className="font-bold">the device used to train the model</span>.
          </p>
        </div>
      </TabsContent>
      
      <TabsContent value="other" className="mt-6">
        <h2 className="text-2xl font-semibold mb-4">Other Visualizations</h2>
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Model Confidence Distributions</CardTitle>
              <CardDescription>Distribution of prediction probabilities across models</CardDescription>
            </CardHeader>
            <CardContent>
              {visualizationPaths.probabilities?.exists ? (
                <div className="relative w-full aspect-video">
                  <Image 
                    src={visualizationPaths.probabilities.path}
                    alt="Probability distributions chart" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              ) : (
                <div className="w-full h-64 bg-gray-100 flex items-center justify-center">
                  <p className="text-gray-400">Probability distributions visualization not available</p>
                </div>
              )}
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Model Agreement</CardTitle>
              <CardDescription>How often models agree on the same prediction</CardDescription>
            </CardHeader>
            <CardContent>
              {visualizationPaths.agreement?.exists ? (
                <div className="relative w-full aspect-video">
                  <Image 
                    src={visualizationPaths.agreement.path}
                    alt="Model agreement chart" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              ) : (
                <div className="w-full h-64 bg-gray-100 flex items-center justify-center">
                  <p className="text-gray-400">Model agreement visualization not available</p>
                </div>
              )}
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader>
              <CardTitle>Prediction Sample</CardTitle>
              <CardDescription>Sample of actual predictions on test matches</CardDescription>
            </CardHeader>
            <CardContent>
              {visualizationPaths.sample?.exists ? (
                <div className="relative w-full aspect-video">
                  <Image 
                    src={visualizationPaths.sample.path}
                    alt="Prediction sample table" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              ) : (
                <div className="w-full h-64 bg-gray-100 flex items-center justify-center">
                  <p className="text-gray-400">Prediction sample visualization not available</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        
        <div className="mt-8 p-4 border rounded-lg bg-gray-50">
          <h3 className="text-lg font-semibold mb-2">About These Visualizations</h3>
          <p className="text-gray-600">
            These visualizations are generated from the model comparison analysis on a test set of Premier League matches. 
            They help illustrate the relative strengths and weaknesses of each prediction model and highlight where they 
            agree or disagree on specific match outcomes. These metrics can differ based on the data used to train the models and <span className="font-bold">the device used to train the model</span>.
          </p>
        </div>
      </TabsContent>
      
      <TabsContent value="model-details" className="mt-6">
        <h2 className="text-2xl font-semibold mb-4">Model-Specific Visualizations</h2>
        
        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <div className={`w-3 h-3 rounded-full bg-purple-500 mr-2`}></div>
            Neural Network Details
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Confusion Matrix</CardTitle>
                <CardDescription>Neural network model predictions vs. actual outcomes</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative w-full aspect-video">
                  <Image 
                    src="/images/visualizations/neural_network/confusion_matrix.png"
                    alt="Neural Network Confusion Matrix" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Training History</CardTitle>
                <CardDescription>Loss and accuracy during model training</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative w-full aspect-video">
                  <Image 
                    src="/images/visualizations/neural_network/training_history.png"
                    alt="Neural Network Training History" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
        
        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <div className={`w-3 h-3 rounded-full bg-green-500 mr-2`}></div>
            Random Forest Details
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Confusion Matrix</CardTitle>
                <CardDescription>Random Forest model predictions vs. actual outcomes</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative w-full aspect-video">
                  <Image 
                    src="/images/visualizations/random_forest/confusion_matrix.png"
                    alt="Random Forest Confusion Matrix" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Feature Importance</CardTitle>
                <CardDescription>Most influential features in Random Forest predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative w-full aspect-video">
                  <Image 
                    src="/images/visualizations/random_forest/feature_importance.png"
                    alt="Random Forest Feature Importance" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
        
        <div className="mb-8">
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <div className={`w-3 h-3 rounded-full bg-blue-500 mr-2`}></div>
            XGBoost Details
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Confusion Matrix</CardTitle>
                <CardDescription>XGBoost model predictions vs. actual outcomes</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative w-full aspect-video">
                  <Image 
                    src="/images/visualizations/xgboost/confusion_matrix.png"
                    alt="XGBoost Confusion Matrix" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Feature Importance</CardTitle>
                <CardDescription>Most influential features in XGBoost predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative w-full aspect-video">
                  <Image 
                    src="/images/visualizations/xgboost/feature_importance.png"
                    alt="XGBoost Feature Importance" 
                    fill
                    style={{ objectFit: 'contain' }}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
        
        <div className="p-4 border rounded-lg bg-gray-50">
          <h3 className="text-lg font-semibold mb-2">About These Visualizations</h3>
          <p className="text-gray-600">
            These detailed model visualizations provide insights into each model&apos;s training process, prediction patterns, 
            and feature importance. The confusion matrices show how each model performs across different match outcomes,
            while the feature importance plots highlight which statistics most influence the predictions.
          </p>
        </div>
      </TabsContent>
    </Tabs>
  );
} 
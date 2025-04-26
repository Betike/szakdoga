"use client";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PredictionForm } from "@/components/PredictionForm";
import { ComparativePrediction } from "@/components/ComparativePrediction";

export default function PredictPage() {
  return (
    <main className="flex flex-col items-center p-6 md:p-10">
      <div className="max-w-5xl w-full">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-3">
            Match Prediction
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Select teams and a prediction model to get match outcome probabilities based on historical performance data.
          </p>
        </div>

        <Tabs defaultValue="single" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="single">Single Model Prediction</TabsTrigger>
            <TabsTrigger value="compare">Compare All Models</TabsTrigger>
          </TabsList>
          
          <TabsContent value="single">
            <div className="p-4 border rounded-lg mt-4">
              <h2 className="text-xl font-semibold mb-4">Single Model Prediction</h2>
              <p className="mb-4 text-gray-600">
                Choose a specific model to generate a prediction for your selected match.
              </p>
              <PredictionForm />
            </div>
          </TabsContent>
          
          <TabsContent value="compare">
            <div className="p-4 border rounded-lg mt-4">
              <h2 className="text-xl font-semibold mb-4">Model Comparison</h2>
              <p className="mb-4 text-gray-600">
                Compare predictions from all available models side by side.
              </p>
              <ComparativePrediction />
            </div>
          </TabsContent>
        </Tabs>
        
        <div className="mt-8 p-4 border rounded-lg bg-gray-50">
          <h2 className="text-xl font-semibold mb-2">About Our Prediction Models</h2>
          <p className="mb-2">
            Our prediction models are trained on historical Premier League match data going back multiple seasons, 
            incorporating team performance statistics, form, and historical match outcomes.
          </p>
          <p>
            The displayed probabilities represent the likelihood of each possible match outcome according to the selected model.
          </p>
        </div>
      </div>
    </main>
  );
}
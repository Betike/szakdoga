"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";

import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { premierLeagueTeams } from "@/data/teams";

// Define the form schema with Zod
const formSchema = z.object({
  homeTeam: z.string().min(1, "Please select a home team"),
  awayTeam: z.string().min(1, "Please select an away team"),
}).refine(data => data.homeTeam !== data.awayTeam, {
  message: "Home team and away team cannot be the same",
  path: ["awayTeam"],
});

type PredictionResult = {
  result: string;
  probabilities: Record<string, number>;
};

// Extended type for ensemble results
type EnsemblePredictionResult = PredictionResult & {
  warning?: string;
  models_used?: number;
  total_models?: number;
  failed_models?: Record<string, string>;
};

type ComparativeResults = Record<string, PredictionResult | EnsemblePredictionResult>;

export function ComparativePrediction() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ComparativeResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Define form
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      homeTeam: "",
      awayTeam: "",
    },
  });

  // Submit handler
  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      setIsLoading(true);
      setError(null);
      setResults(null);

      // Fetch predictions from all models
      const predictorTypes = ["xgboost", "random_forest", "pytorch", "ensemble"];
      const allResults: ComparativeResults = {};

      await Promise.all(
        predictorTypes.map(async (predictorType) => {
          const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              ...values,
              predictorType,
            }),
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Failed to make prediction with ${predictorType}`);
          }

          const result = await response.json();
          allResults[predictorType] = result;
        })
      );

      setResults(allResults);
    } catch (error: unknown) {
      setError(error instanceof Error ? error.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  // Helper function to format result
  const formatResult = (result: string) => {
    switch (result) {
      case "H":
        return "Home Win";
      case "A":
        return "Away Win";
      case "D":
        return "Draw";
      default:
        return result;
    }
  };

  // Helper function to get model name
  const getModelName = (key: string) => {
    switch (key) {
      case "xgboost":
        return "XGBoost";
      case "random_forest":
        return "Random Forest";
      case "pytorch":
        return "PyTorch NN";
      case "ensemble":
        return "Ensemble";
      default:
        return key;
    }
  };

  // Helper function to get color based on prediction
  const getResultColor = (result: string) => {
    switch (result) {
      case "H":
        return "bg-green-100 text-green-800";
      case "A":
        return "bg-blue-100 text-blue-800";
      case "D":
        return "bg-yellow-100 text-yellow-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      <Card>


        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="homeTeam"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Home Team</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select home team" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {premierLeagueTeams.map((team) => (
                            <SelectItem key={team} value={team}>
                              {team}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="awayTeam"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Away Team</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select away team" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {premierLeagueTeams.map((team) => (
                            <SelectItem key={team} value={team}>
                              {team}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <Button type="submit" className="w-full" disabled={isLoading}>
                {isLoading ? "Running all models..." : "Compare All Models"}
              </Button>
            </form>
          </Form>
        </CardContent>

        {error && (
          <div className="px-6 pb-6">
            <div className="p-4 bg-red-50 text-red-600 rounded-md">
              {error}
            </div>
          </div>
        )}

        {results && (
          <CardFooter className="flex flex-col">
            <div className="w-full border-t pt-4">
              <h3 className="text-lg font-semibold mb-4">Comparison Results</h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(results).map(([modelKey, result]) => {
                  const isEnsemble = modelKey === 'ensemble';
                  const ensembleResult = isEnsemble ? result as EnsemblePredictionResult : null;
                  
                  return (
                    <Card key={modelKey} className="overflow-hidden">
                      <CardHeader className="p-4 pb-2">
                        <CardTitle className="text-base">{getModelName(modelKey)}</CardTitle>
                        <div className={`inline-block px-2 py-1 rounded text-sm font-medium ${getResultColor(result.result)}`}>
                          {formatResult(result.result)}
                        </div>
                      </CardHeader>
                      <CardContent className="p-4 pt-2">
                        {isEnsemble && ensembleResult?.warning && (
                          <div className="p-2 mb-3 bg-yellow-50 border border-yellow-200 rounded-md text-yellow-700 text-xs">
                            <p className="font-medium">Warning</p>
                            <p>{ensembleResult.warning}</p>
                          </div>
                        )}
                        <div className="space-y-2">
                          {Object.entries(result.probabilities)
                            .sort(([, a], [, b]) => b - a)
                            .map(([outcome, probability]) => (
                              <div key={outcome} className="grid grid-cols-5 gap-2 items-center">
                                <span className="col-span-2 text-sm">{outcome}</span>
                                <div className="col-span-2 h-2 bg-gray-200 rounded-full">
                                  <div
                                    className="h-full bg-blue-600 rounded-full"
                                    style={{ width: `${probability * 100}%` }}
                                  />
                                </div>
                                <span className="text-xs text-right">{(probability * 100).toFixed(1)}%</span>
                              </div>
                            ))}
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
              
              <div className="mt-6 p-4 bg-gray-50 rounded-md">
                <h4 className="font-semibold mb-2">Prediction Agreement</h4>
                <p className="text-sm text-gray-600">
                  {Object.values(results).every(r => r.result === Object.values(results)[0].result)
                    ? "All models agree on the prediction! In agreement the accuracy of the predictions is 65%."
                    : "Models have different predictions. Consider using the Ensemble model."}
                </p>
              </div>
            </div>
          </CardFooter>
        )}
      </Card>
    </div>
  );
} 
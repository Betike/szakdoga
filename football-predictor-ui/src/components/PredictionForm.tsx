"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";

import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { premierLeagueTeams, predictorOptions } from "@/data/teams";

// Define the form schema with Zod
const formSchema = z.object({
  predictorType: z.string().min(1, "Please select a predictor"),
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

export function PredictionForm() {
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<EnsemblePredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Define form
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      predictorType: "",
      homeTeam: "",
      awayTeam: "",
    },
  });

  // Submit handler
  const onSubmit = async (values: z.infer<typeof formSchema>) => {
    try {
      setIsLoading(true);
      setError(null);
      setPredictionResult(null);

      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(values),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to make prediction");
      }

      const result = await response.json();
      setPredictionResult(result);
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

  return (
    <div className="w-full max-w-3xl mx-auto">
      <Card>

        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
              <FormField
                control={form.control}
                name="predictorType"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Prediction Model</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a prediction model" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {predictorOptions.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />

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
                {isLoading ? "Predicting..." : "Predict Match Outcome"}
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

        {predictionResult && (
          <CardFooter className="flex flex-col">
            <div className="w-full border-t pt-4">
              <h3 className="text-lg font-semibold mb-2">
                Prediction Result: {formatResult(predictionResult.result)}
              </h3>

              {predictionResult.warning && (
                <div className="p-3 mb-4 bg-yellow-50 border border-yellow-200 rounded-md text-yellow-700">
                  <p className="font-medium">Warning</p>
                  <p className="text-sm">{predictionResult.warning}</p>
                  {predictionResult.failed_models && (
                    <div className="mt-2 text-sm">
                      <p>Failed models:</p>
                      <ul className="list-disc ml-5">
                        {Object.entries(predictionResult.failed_models).map(([model, error]) => (
                          <li key={model}>{model}: {error}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              <div className="space-y-2 w-full">
                {Object.entries(predictionResult.probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([outcome, probability]) => (
                    <div key={outcome} className="flex justify-between">
                      <span>{outcome}</span>
                      <div className="flex items-center">
                        <div className="w-48 h-2 bg-gray-200 rounded-full mr-2">
                          <div
                            className="h-full bg-blue-600 rounded-full"
                            style={{ width: `${probability * 100}%` }}
                          />
                        </div>
                        <span className="text-sm">{(probability * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          </CardFooter>
        )}
      </Card>
    </div>
  );
} 
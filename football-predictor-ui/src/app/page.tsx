import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import Link from "next/link";
//import Image from "next/image";

export default function Home() {
  return (
    <main className="flex flex-col items-center p-6 md:p-10">
      <div className="max-w-5xl w-full">
        {/* Hero section */}
        <div className="py-10 md:py-16 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Premier League Football Predictor
          </h1>
          <p className="text-lg text-gray-600 mb-6 max-w-2xl mx-auto">
            Predict match outcomes using advanced machine learning models trained on historical Premier League data
          </p>
          <div className="flex justify-center gap-4">
            <Button asChild size="lg">
              <Link href="/predict">Make a Prediction</Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link href="/visualizations">View Model Performance</Link>
            </Button>
          </div>
        </div>

        {/* Features section */}
        <div className="py-10">
          <h2 className="text-2xl font-bold mb-6 text-center">About Our Predictive Models</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>XGBoost Model</CardTitle>
                <CardDescription>Gradient Boosting</CardDescription>
              </CardHeader>
              <CardContent>
                <p>
                  Our XGBoost model provides predictions with {Math.round(58.17 * 100) / 100}% accuracy on test data, 
                  with strong performance on predicting home and away wins.
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Random Forest</CardTitle>
                <CardDescription>Ensemble Learning</CardDescription>
              </CardHeader>
              <CardContent>
                <p>
                  The Random Forest model achieves {Math.round(60.56 * 100) / 100}% accuracy with balanced 
                  predictions across all match outcomes.
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Neural Network</CardTitle>
                <CardDescription>Deep Learning with PyTorch</CardDescription>
              </CardHeader>
              <CardContent>
                <p>
                  Our deep learning model is trained on extensive feature sets with batch normalization
                  and dropout layers to prevent overfitting.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
        
        {/* How it works section */}
        <div className="py-10">
          <h2 className="text-2xl font-bold mb-6 text-center">How It Works</h2>
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row gap-6 items-center">
              <div className="md:w-1/2">
                <h3 className="text-xl font-semibold mb-2">Data Collection & Processing</h3>
                <p className="text-gray-600">
                  We&apos;ve collected thousands of Premier League matches, extracting key statistics
                  and processing them to create informative features for our models.
                </p>
              </div>
              <div className="md:w-1/2 bg-white h-40 rounded-lg flex items-center justify-center">
                <div className="w-full h-full flex items-center justify-center" dangerouslySetInnerHTML={{ __html: `
                    <svg width="240" height="160" viewBox="0 0 240 160" xmlns="http://www.w3.org/2000/svg">
                      <!-- Háttér -->
                      <rect x="0" y="0" width="240" height="160" fill="#f0f8ff" rx="8" ry="8" />
                      
                      <!-- Cím -->
                      <text x="120" y="25" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" font-weight="bold" fill="#033C73">Data processing</text>
                      
                      <!-- Nyers adatok -->
                      <rect x="30" y="45" width="40" height="50" fill="#ffffff" stroke="#033C73" stroke-width="2" rx="4" ry="4" />
                      <rect x="34" y="50" width="32" height="5" fill="#033C73" opacity="0.3" rx="1" ry="1" />
                      <rect x="34" y="60" width="32" height="5" fill="#033C73" opacity="0.3" rx="1" ry="1" />
                      <rect x="34" y="70" width="32" height="5" fill="#033C73" opacity="0.3" rx="1" ry="1" />
                      <rect x="34" y="80" width="20" height="5" fill="#033C73" opacity="0.3" rx="1" ry="1" />
                      <text x="50" y="105" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#033C73">Raw data</text>
                      
                      <!-- Nyíl -->
                      <line x1="80" y1="70" x2="110" y2="70" stroke="#033C73" stroke-width="2" />
                      <polygon points="108,65 115,70 108,75" fill="#033C73" />
                      
                      <!-- Feldolgozás szimbólum (fogaskerék) -->
                      <circle cx="120" cy="70" r="20" fill="#ffffff" stroke="#033C73" stroke-width="2" />
                      <circle cx="120" cy="70" r="12" fill="none" stroke="#033C73" stroke-width="1" />
                      <!-- Fogaskerék fogai -->
                      <line x1="120" y1="50" x2="120" y2="55" stroke="#033C73" stroke-width="2" />
                      <line x1="120" y1="85" x2="120" y2="90" stroke="#033C73" stroke-width="2" />
                      <line x1="100" y1="70" x2="105" y2="70" stroke="#033C73" stroke-width="2" />
                      <line x1="135" y1="70" x2="140" y2="70" stroke="#033C73" stroke-width="2" />
                      <line x1="106" y1="56" x2="110" y2="60" stroke="#033C73" stroke-width="2" />
                      <line x1="130" y1="80" x2="134" y2="84" stroke="#033C73" stroke-width="2" />
                      <line x1="106" y1="84" x2="110" y2="80" stroke="#033C73" stroke-width="2" />
                      <line x1="130" y1="60" x2="134" y2="56" stroke="#033C73" stroke-width="2" />
                      <text x="120" y="105" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#033C73">Processing</text>
                      
                      <!-- Nyíl -->
                      <line x1="150" y1="70" x2="180" y2="70" stroke="#033C73" stroke-width="2" />
                      <polygon points="178,65 185,70 178,75" fill="#033C73" />
                      
                      <!-- Feldolgozott adatok -->
                      <rect x="170" y="45" width="40" height="50" fill="#ffffff" stroke="#033C73" stroke-width="2" rx="4" ry="4" />
                      <rect x="174" y="50" width="32" height="5" fill="#0066CC" opacity="0.5" rx="1" ry="1" />
                      <rect x="174" y="60" width="32" height="5" fill="#0066CC" opacity="0.5" rx="1" ry="1" />
                      <rect x="174" y="70" width="32" height="5" fill="#0066CC" opacity="0.5" rx="1" ry="1" />
                      <rect x="174" y="80" width="20" height="5" fill="#0066CC" opacity="0.5" rx="1" ry="1" />
                      <text x="190" y="105" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#033C73">Clean data</text>
                      
                    </svg> 
                ` }} />
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-6 items-center">
              <div className="md:w-1/2 md:order-2">
                <h3 className="text-xl font-semibold mb-2">Model Training & Evaluation</h3>
                <p className="text-gray-600">
                  Our models are trained on historical data and evaluated using chronological
                  validation to ensure they perform well on future matches.
                </p>
              </div>
              <div className="md:w-1/2 md:order-1 bg-white h-40 rounded-lg flex items-center justify-center">
                <div className="w-full h-full flex items-center justify-center" dangerouslySetInnerHTML={{ __html: `
                  <svg width="240" height="160" viewBox="0 0 240 160" xmlns="http://www.w3.org/2000/svg">
                    <rect x="0" y="0" width="240" height="160" fill="#f0f8ff" rx="8" ry="8" />
                    
                    <text x="120" y="25" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" font-weight="bold" fill="#033C73">Model training</text>
                    
                    <rect x="20" y="45" width="60" height="40" fill="#ffffff" stroke="#033C73" stroke-width="2" rx="4" ry="4" />
                    <rect x="25" y="50" width="50" height="5" fill="#0066CC" opacity="0.5" rx="1" ry="1" />
                    <rect x="25" y="60" width="50" height="5" fill="#0066CC" opacity="0.5" rx="1" ry="1" />
                    <rect x="25" y="70" width="35" height="5" fill="#0066CC" opacity="0.5" rx="1" ry="1" />
                    <text x="50" y="95" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#033C73">Training data</text>
                    
                    <line x1="90" y1="65" x2="110" y2="65" stroke="#033C73" stroke-width="2" />
                    <polygon points="108,60 115,65 108,70" fill="#033C73" />
                    
                    <rect x="95" y="35" width="50" height="60" fill="#ffffff" stroke="#033C73" stroke-width="2" rx="10" ry="10" />
                    
                    <path d="M100,60 Q120,30 140,60" fill="none" stroke="#033C73" stroke-width="2" />
                    <path d="M100,70 Q120,40 140,70" fill="none" stroke="#033C73" stroke-width="2" />
                    <path d="M100,80 Q120,50 140,80" fill="none" stroke="#033C73" stroke-width="2" />
                    <text x="120" y="105" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#033C73">Training process</text>
                    
                    <line x1="150" y1="65" x2="170" y2="65" stroke="#033C73" stroke-width="2" />
                    <polygon points="168,60 175,65 168,70" fill="#033C73" />
                    
                    <rect x="160" y="45" width="60" height="40" fill="#ffffff" stroke="#033C73" stroke-width="2" rx="4" ry="4" />
                    <circle cx="190" cy="65" r="15" fill="none" stroke="#033C73" stroke-width="2" />
                    <circle cx="190" cy="65" r="10" fill="#0066CC" opacity="0.2" />
                    <circle cx="190" cy="65" r="5" fill="#0066CC" opacity="0.4" />
                    <text x="190" y="95" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#033C73">Trained model</text>
                    
                  </svg> 
                ` }} />
              </div>
            </div>
            
            <div className="flex flex-col md:flex-row gap-6 items-center">
              <div className="md:w-1/2">
                <h3 className="text-xl font-semibold mb-2">Prediction & Analysis</h3>
                <p className="text-gray-600">
                  Select teams, choose your model, and get detailed predictions with
                  probability distributions for each possible match outcome.
                </p>
              </div>
              <div className="md:w-1/2 bg-white h-40 rounded-lg flex items-center justify-center">
                <div className="w-full h-full flex items-center justify-center" dangerouslySetInnerHTML={{ __html: `
                  <svg width="240" height="160" viewBox="0 0 240 160" xmlns="http://www.w3.org/2000/svg">
                    <rect x="0" y="0" width="240" height="160" fill="#f0f8ff" rx="8" ry="8" />
                    
                    <text x="120" y="25" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" font-weight="bold" fill="#033C73">Prediction</text>
                    
                    <rect x="15" y="40" width="90" height="30" fill="#ffffff" stroke="#033C73" stroke-width="2" rx="4" ry="4" />
                    <text x="60" y="60" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#033C73">Arsenal/Chelsea</text>
                    
                    <line x1="60" y1="85" x2="60" y2="100" stroke="#033C73" stroke-width="2" />
                    <polygon points="55,98 60,105 65,98" fill="#033C73" />
                    
                    <rect x="15" y="105" width="90" height="30" fill="#ffffff" stroke="#033C73" stroke-width="2" rx="4" ry="4" />
                    <circle cx="35" cy="120" r="8" fill="none" stroke="#033C73" stroke-width="1.5" />
                    <circle cx="35" cy="120" r="4" fill="#0066CC" opacity="0.3" />
                    <text x="60" y="124" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#033C73">Model</text>
                    
                    <line x1="115" y1="120" x2="135" y2="120" stroke="#033C73" stroke-width="2" />
                    <polygon points="133,115 140,120 133,125" fill="#033C73" />
                    
                    <rect x="135" y="60" width="90" height="80" fill="#ffffff" stroke="#033C73" stroke-width="2" rx="4" ry="4" />
                    
                    <text x="180" y="75" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" font-weight="bold" fill="#033C73">Result</text>
                    
                    <rect x="145" y="85" width="70" height="10" fill="#e6e6e6" rx="3" ry="3" />
                    <rect x="145" y="85" width="45" height="10" fill="#0066CC" opacity="0.6" rx="3" ry="3" />
                    <text x="150" y="93" font-family="Arial, sans-serif" font-size="8" fill="#033C73">Home: 44%</text>
                    
                    <rect x="145" y="100" width="70" height="10" fill="#e6e6e6" rx="3" ry="3" />
                    <rect x="145" y="100" width="20" height="10" fill="#0066CC" opacity="0.6" rx="3" ry="3" />
                    <text x="150" y="108" font-family="Arial, sans-serif" font-size="8" fill="#033C73">Draw: 34%</text>
                    
                    <rect x="145" y="115" width="70" height="10" fill="#e6e6e6" rx="3" ry="3" />
                    <rect x="145" y="115" width="10" height="10" fill="#0066CC" opacity="0.6" rx="3" ry="3" />
                    <text x="150" y="123" font-family="Arial, sans-serif" font-size="8" fill="#033C73">Away: 23%</text>
                    
                    <text x="180" y="135" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" font-weight="bold" fill="#033C73">Home win</text>
                    
                  </svg> 
                ` }} />
              </div>
            </div>
          </div>
        </div>
        </div>
      </main>
  );
}

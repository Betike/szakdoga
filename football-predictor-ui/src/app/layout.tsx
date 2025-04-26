import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Navigation } from "@/components/Navigation";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Premier League Football Predictor",
  description: "Predict football match outcomes using machine learning models including XGBoost, Random Forest, and PyTorch",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <div className="min-h-screen flex flex-col">
          <Navigation />
          <div className="flex-1">{children}</div>
          <footer className="border-t py-6 text-center text-gray-500 text-xs">
            <p>
              Premier League Football Predictor - Uses multiple machine learning models to predict match outcomes
            </p>
            <p className="mt-2">
              Developed with Next.js, React, and Shadcn UI components
            </p>
          </footer>
        </div>
      </body>
    </html>
  );
}

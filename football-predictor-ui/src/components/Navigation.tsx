"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

export function Navigation() {
  const pathname = usePathname();
  
  const links = [
    { href: "/", label: "Home" },
    { href: "/predict", label: "Prediction" },
    { href: "/visualizations", label: "Visualizations" },
  ];

  return (
    <div className="border-b">
      <div className="flex h-16 items-center px-4 max-w-5xl mx-auto">
        <div className="mr-4 hidden md:flex">
          <Link href="/" className="font-bold text-xl">Premier League Predictor</Link>
        </div>
        <nav className="flex items-center space-x-4 lg:space-x-6 ml-auto">
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "text-sm font-medium transition-colors hover:text-primary",
                pathname === link.href
                  ? "text-foreground"
                  : "text-muted-foreground"
              )}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      </div>
    </div>
  );
} 
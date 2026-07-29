"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard, Upload, ScanLine, History,
  BarChart2, Settings, Activity, Moon, Sun, Microscope,
} from "lucide-react";
import { useTheme } from "next-themes";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/upload", label: "Upload Image", icon: Upload },
  { href: "/results", label: "Detection Results", icon: ScanLine },
  { href: "/history", label: "Prediction History", icon: History },
  { href: "/metrics", label: "Training Metrics", icon: BarChart2 },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <aside className="w-64 border-r border-border flex flex-col bg-card shrink-0">
      <div className="p-5 border-b border-border">
        <div className="flex items-center gap-2">
          <Activity className="text-primary w-6 h-6" />
          <div>
            <p className="font-bold text-sm leading-tight">Blood Cell</p>
            <p className="text-xs text-muted-foreground">Detection Agent</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {NAV.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
              pathname === href
                ? "bg-primary/10 text-primary font-medium"
                : "text-muted-foreground hover:bg-accent hover:text-foreground"
            )}
          >
            <Icon className="w-4 h-4 shrink-0" />
            {label}
          </Link>
        ))}
      </nav>

      <div className="p-3 border-t border-border">
        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="flex items-center gap-3 px-3 py-2 rounded-md text-sm text-muted-foreground hover:bg-accent hover:text-foreground w-full transition-colors"
        >
          {mounted && (theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />)}
          {mounted && (theme === "dark" ? "Light Mode" : "Dark Mode")}
        </button>
      </div>
    </aside>
  );
}

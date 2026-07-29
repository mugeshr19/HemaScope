"use client";
import { useEffect, useState } from "react";
import { getMetrics, getHealth } from "@/lib/api";
import type { Metrics, HealthStatus } from "@/types";
import { StatCard } from "@/components/stat-card";
import { Activity, Cpu, Database, Microscope, CheckCircle, AlertCircle } from "lucide-react";

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    getMetrics().then(setMetrics).catch(console.error);
    getHealth().then(setHealth).catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground text-sm mt-1">Blood Cell Detection Agent — Overview</p>
      </div>

      {/* Health Status */}
      <div className="flex items-center gap-3 p-3 rounded-lg border border-border bg-card">
        {health?.status === "healthy" ? (
          <CheckCircle className="w-5 h-5 text-green-400 shrink-0" />
        ) : (
          <AlertCircle className="w-5 h-5 text-yellow-400 shrink-0" />
        )}
        <div className="flex gap-6 text-sm">
          <span>
            Status: <span className={health?.status === "healthy" ? "text-green-400" : "text-yellow-400"}>
              {health?.status ?? "—"}
            </span>
          </span>
          <span>
            Model: <span className={health?.model_loaded ? "text-green-400" : "text-red-400"}>
              {health?.model_loaded ? "Loaded" : "Not Loaded"}
            </span>
          </span>
          <span>
            Database: <span className={health?.database_connected ? "text-green-400" : "text-red-400"}>
              {health?.database_connected ? "Connected" : "Disconnected"}
            </span>
          </span>
          <span className="text-muted-foreground">v{health?.version}</span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        <StatCard
          title="Total Predictions"
          value={metrics?.total_predictions ?? 0}
          icon={<Microscope className="w-4 h-4" />}
        />
        <StatCard
          title="Total Cells Detected"
          value={metrics?.total_cells_detected ?? 0}
          icon={<Activity className="w-4 h-4" />}
        />
        <StatCard
          title="Avg Inference Time"
          value={metrics ? `${metrics.avg_inference_time}s` : "—"}
          icon={<Cpu className="w-4 h-4" />}
        />
        <StatCard title="Avg RBC / Image" value={metrics?.avg_rbc_per_image ?? 0} color="red" />
        <StatCard title="Avg WBC / Image" value={metrics?.avg_wbc_per_image ?? 0} color="green" />
        <StatCard title="Avg Platelet / Image" value={metrics?.avg_platelet_per_image ?? 0} color="blue" />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 gap-4">
        <a href="/upload" className="p-5 rounded-lg border border-border bg-card hover:bg-accent transition-colors">
          <p className="font-semibold">Run Detection</p>
          <p className="text-sm text-muted-foreground mt-1">Upload a blood smear image for analysis</p>
        </a>
        <a href="/history" className="p-5 rounded-lg border border-border bg-card hover:bg-accent transition-colors">
          <p className="font-semibold">View History</p>
          <p className="text-sm text-muted-foreground mt-1">Browse all previous predictions</p>
        </a>
      </div>
    </div>
  );
}

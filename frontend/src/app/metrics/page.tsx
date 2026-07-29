"use client";
import { useEffect, useState } from "react";
import { getMetrics } from "@/lib/api";
import type { Metrics } from "@/types";
import { StatCard } from "@/components/stat-card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { CLASS_COLORS } from "@/lib/utils";

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  useEffect(() => {
    getMetrics().then(setMetrics).catch(console.error);
  }, []);

  const avgData = metrics
    ? [
        { name: "RBC", avg: metrics.avg_rbc_per_image, fill: CLASS_COLORS.RBC },
        { name: "WBC", avg: metrics.avg_wbc_per_image, fill: CLASS_COLORS.WBC },
        { name: "Platelet", avg: metrics.avg_platelet_per_image, fill: CLASS_COLORS.Platelet },
      ]
    : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Metrics</h1>
        <p className="text-muted-foreground text-sm mt-1">Aggregate inference statistics</p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <StatCard title="Total Predictions" value={metrics?.total_predictions ?? 0} />
        <StatCard title="Total Cells Detected" value={metrics?.total_cells_detected ?? 0} />
        <StatCard title="Avg Inference Time" value={metrics ? `${metrics.avg_inference_time}s` : "—"} />
      </div>

      <div className="rounded-lg border border-border bg-card p-5">
        <p className="text-sm font-medium mb-4">Average Cell Count per Image</p>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={avgData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 47% 20%)" />
            <XAxis dataKey="name" tick={{ fontSize: 12, fill: "hsl(215 20% 65%)" }} />
            <YAxis tick={{ fontSize: 12, fill: "hsl(215 20% 65%)" }} />
            <Tooltip
              contentStyle={{ background: "hsl(222 47% 14%)", border: "1px solid hsl(222 47% 20%)", borderRadius: 6 }}
            />
            <Bar dataKey="avg" radius={[4, 4, 0, 0]}>
              {avgData.map((entry, i) => (
                <rect key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="rounded-lg border border-border bg-card p-5">
        <p className="text-sm font-medium mb-3">Training Notes</p>
        <p className="text-sm text-muted-foreground">
          Training metrics (loss curves, PR curves, confusion matrix) are generated during training and saved to{" "}
          <code className="text-xs bg-muted px-1 py-0.5 rounded">runs/train/blood_cell_detection/</code>.
          View them with TensorBoard:
        </p>
        <pre className="mt-3 text-xs bg-muted/50 rounded p-3 overflow-x-auto">
          tensorboard --logdir runs/train/blood_cell_detection
        </pre>
      </div>
    </div>
  );
}

"use client";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { CLASS_COLORS } from "@/lib/utils";
import type { Detection } from "@/types";

interface ConfidenceChartProps {
  detections: Detection[];
}

export function ConfidenceChart({ detections }: ConfidenceChartProps) {
  const buckets = Array.from({ length: 10 }, (_, i) => ({
    range: `${(i * 10).toString().padStart(2, "0")}-${((i + 1) * 10).toString().padStart(2, "0")}%`,
    RBC: 0, WBC: 0, Platelet: 0,
  }));

  detections.forEach((d) => {
    const idx = Math.min(Math.floor(d.confidence * 10), 9);
    (buckets[idx] as Record<string, number>)[d.class]++;
  });

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={buckets} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(222 47% 20%)" />
        <XAxis dataKey="range" tick={{ fontSize: 10, fill: "hsl(215 20% 65%)" }} />
        <YAxis tick={{ fontSize: 10, fill: "hsl(215 20% 65%)" }} />
        <Tooltip
          contentStyle={{ background: "hsl(222 47% 14%)", border: "1px solid hsl(222 47% 20%)", borderRadius: 6 }}
        />
        <Bar dataKey="RBC" fill={CLASS_COLORS.RBC} stackId="a" />
        <Bar dataKey="WBC" fill={CLASS_COLORS.WBC} stackId="a" />
        <Bar dataKey="Platelet" fill={CLASS_COLORS.Platelet} stackId="a" />
      </BarChart>
    </ResponsiveContainer>
  );
}

"use client";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { CLASS_COLORS } from "@/lib/utils";

interface CellDistributionChartProps {
  rbc: number;
  wbc: number;
  platelet: number;
}

export function CellDistributionChart({ rbc, wbc, platelet }: CellDistributionChartProps) {
  const data = [
    { name: "RBC", value: rbc },
    { name: "WBC", value: wbc },
    { name: "Platelet", value: platelet },
  ].filter((d) => d.value > 0);

  if (data.length === 0) return <p className="text-muted-foreground text-sm text-center py-8">No data</p>;

  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie data={data} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={3} dataKey="value">
          {data.map((entry) => (
            <Cell key={entry.name} fill={CLASS_COLORS[entry.name]} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{ background: "hsl(222 47% 14%)", border: "1px solid hsl(222 47% 20%)", borderRadius: 6 }}
          labelStyle={{ color: "hsl(213 31% 91%)" }}
        />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}

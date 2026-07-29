import { cn } from "@/lib/utils";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  color?: "red" | "green" | "blue" | "default";
  icon?: React.ReactNode;
}

const colorMap = {
  red: "border-red-500/30 bg-red-500/5",
  green: "border-green-500/30 bg-green-500/5",
  blue: "border-blue-500/30 bg-blue-500/5",
  default: "border-border bg-card",
};

const valueColorMap = {
  red: "text-red-400",
  green: "text-green-400",
  blue: "text-blue-400",
  default: "text-foreground",
};

export function StatCard({ title, value, subtitle, color = "default", icon }: StatCardProps) {
  return (
    <div className={cn("rounded-lg border p-4", colorMap[color])}>
      <div className="flex items-start justify-between">
        <p className="text-xs text-muted-foreground uppercase tracking-wider">{title}</p>
        {icon && <span className="text-muted-foreground">{icon}</span>}
      </div>
      <p className={cn("text-3xl font-bold mt-2", valueColorMap[color])}>{value}</p>
      {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
    </div>
  );
}

"use client";
import { useState } from "react";
import axios from "axios";
import { Dropzone } from "@/components/dropzone";
import { StatCard } from "@/components/stat-card";
import { toast } from "sonner";
import { Loader2, ShieldAlert, ShieldCheck, ShieldQuestion, Activity } from "lucide-react";

interface CellPrediction {
  cell_index: number;
  label: "Parasitized" | "Uninfected";
  p_infected: number;
  confidence: number;
}

interface MalariaResult {
  total_rbc: number;
  infected_rbc: number;
  parasite_density_pct: number;
  confidence: number;
  risk_level: "Negative" | "Low" | "Moderate" | "High";
  recommendation: string;
  per_cell_predictions: CellPrediction[];
  agent1: { total_cells: number; rbc: number; wbc: number; platelet: number };
}

const RISK_STYLES: Record<string, { color: string; icon: React.ReactNode }> = {
  Negative: { color: "text-green-400 border-green-500/30 bg-green-500/5",  icon: <ShieldCheck className="w-6 h-6 text-green-400" /> },
  Low:      { color: "text-yellow-400 border-yellow-500/30 bg-yellow-500/5", icon: <ShieldQuestion className="w-6 h-6 text-yellow-400" /> },
  Moderate: { color: "text-orange-400 border-orange-500/30 bg-orange-500/5", icon: <ShieldAlert className="w-6 h-6 text-orange-400" /> },
  High:     { color: "text-red-400 border-red-500/30 bg-red-500/5",         icon: <ShieldAlert className="w-6 h-6 text-red-400" /> },
};

export default function MalariaPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<MalariaResult | null>(null);

  const handleRun = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
      const { data } = await axios.post<MalariaResult>(`${base}/malaria`, form);
      setResult(data);
      toast.success(`Screening complete — Risk: ${data.risk_level}`);
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : "Screening failed");
    } finally {
      setLoading(false);
    }
  };

  const risk = result ? (RISK_STYLES[result.risk_level] ?? RISK_STYLES.Negative) : null;
  const infected = result?.per_cell_predictions.filter((c) => c.label === "Parasitized") ?? [];
  const uninfected = result?.per_cell_predictions.filter((c) => c.label === "Uninfected") ?? [];

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Activity className="w-6 h-6" /> Malaria Screening — Agent 2
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          DenseNet-121 · 94.92% accuracy · NIH Malaria Dataset · For research use only
        </p>
      </div>

      {/* Upload */}
      <div className="rounded-lg border border-border bg-card p-6 space-y-4">
        <Dropzone onFile={setFile} disabled={loading} />
        <button
          onClick={handleRun}
          disabled={!file || loading}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-md bg-primary text-primary-foreground font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary/90 transition-colors"
        >
          {loading ? (
            <><Loader2 className="w-4 h-4 animate-spin" /> Screening RBC Crops...</>
          ) : (
            <><Activity className="w-4 h-4" /> Run Malaria Screening</>
          )}
        </button>
      </div>

      {result && (
        <>
          {/* Risk Banner */}
          <div className={`rounded-lg border p-4 flex items-center gap-4 ${risk?.color}`}>
            {risk?.icon}
            <div>
              <p className="font-bold text-lg">{result.risk_level} Risk</p>
              <p className="text-sm mt-0.5 opacity-90">{result.recommendation}</p>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard title="Total RBC Screened" value={result.total_rbc} icon={<Activity className="w-4 h-4" />} />
            <StatCard title="Infected RBC" value={result.infected_rbc} color="red" />
            <StatCard title="Parasite Density" value={`${result.parasite_density_pct}%`} color={result.parasite_density_pct > 5 ? "red" : result.parasite_density_pct > 0 ? "default" : "green"} />
            <StatCard title="Model Confidence" value={`${(result.confidence * 100).toFixed(1)}%`} color="blue" />
          </div>

          {/* Agent 1 summary */}
          <div className="rounded-lg border border-border bg-card p-4">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Agent 1 — Cell Detection Summary</p>
            <div className="grid grid-cols-4 gap-3 text-center text-sm">
              {[
                { label: "Total Cells", value: result.agent1.total_cells },
                { label: "RBC", value: result.agent1.rbc, cls: "text-red-400" },
                { label: "WBC", value: result.agent1.wbc, cls: "text-green-400" },
                { label: "Platelets", value: result.agent1.platelet, cls: "text-blue-400" },
              ].map(({ label, value, cls }) => (
                <div key={label}>
                  <p className="text-muted-foreground text-xs">{label}</p>
                  <p className={`text-xl font-bold ${cls ?? ""}`}>{value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Per-cell table */}
          <div className="rounded-lg border border-border overflow-hidden">
            <div className="px-4 py-3 bg-muted/30 flex items-center justify-between">
              <p className="text-sm font-medium">Per-Cell Predictions ({result.per_cell_predictions.length})</p>
              <div className="flex gap-3 text-xs text-muted-foreground">
                <span className="text-red-400 font-medium">{infected.length} Parasitized</span>
                <span className="text-green-400 font-medium">{uninfected.length} Uninfected</span>
              </div>
            </div>
            <div className="overflow-auto max-h-72">
              <table className="w-full text-sm">
                <thead className="bg-muted/20 sticky top-0">
                  <tr>
                    {["Cell #", "Label", "P(Infected)", "Confidence"].map((h) => (
                      <th key={h} className="px-4 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {result.per_cell_predictions.map((c) => (
                    <tr key={c.cell_index} className="hover:bg-accent/20">
                      <td className="px-4 py-2 font-mono text-xs">{c.cell_index + 1}</td>
                      <td className="px-4 py-2">
                        <span className={`font-medium ${c.label === "Parasitized" ? "text-red-400" : "text-green-400"}`}>
                          {c.label}
                        </span>
                      </td>
                      <td className="px-4 py-2">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                            <div
                              className={`h-full rounded-full ${c.p_infected > 0.4 ? "bg-red-400" : "bg-green-400"}`}
                              style={{ width: `${c.p_infected * 100}%` }}
                            />
                          </div>
                          <span className="text-xs w-10 text-right">{(c.p_infected * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-2 text-xs">{(c.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <p className="text-xs text-muted-foreground">
            ⚠️ For research use only. Not a medical diagnostic tool. Clinical review required before any diagnostic use.
          </p>
        </>
      )}
    </div>
  );
}

"use client";
import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import axios from "axios";
import { FlaskConical, Loader2, ShieldAlert, ShieldCheck, ShieldQuestion, AlertTriangle } from "lucide-react";
import { StatCard } from "@/components/stat-card";

interface CellPrediction {
  cell_index: number;
  label: "Normal" | "Sickle" | "Crescent" | "Elongated";
  confidence: number;
  probabilities: Record<string, number>;
  crop_url?: string;
}

interface MorphologyResult {
  total_rbc: number;
  normal_count: number;
  abnormal_count: number;
  abnormal_pct: number;
  class_counts: Record<string, number>;
  severity: "Normal" | "Mild" | "Moderate" | "Severe";
  recommendation: string;
  per_cell_predictions: CellPrediction[];
}

const SEVERITY_STYLES: Record<string, { color: string; icon: React.ReactNode }> = {
  Normal:   { color: "text-green-400 border-green-500/30 bg-green-500/5",   icon: <ShieldCheck className="w-6 h-6 text-green-400" /> },
  Mild:     { color: "text-yellow-400 border-yellow-500/30 bg-yellow-500/5", icon: <ShieldQuestion className="w-6 h-6 text-yellow-400" /> },
  Moderate: { color: "text-orange-400 border-orange-500/30 bg-orange-500/5", icon: <ShieldAlert className="w-6 h-6 text-orange-400" /> },
  Severe:   { color: "text-red-400 border-red-500/30 bg-red-500/5",          icon: <AlertTriangle className="w-6 h-6 text-red-400" /> },
};

const LABEL_COLORS: Record<string, string> = {
  Normal:   "text-green-400",
  Sickle:   "text-red-400",
  Crescent: "text-orange-400",
  Elongated:"text-yellow-400",
};

function MorphologyContent() {
  const params = useSearchParams();
  const id = params.get("id");
  const [result, setResult] = useState<MorphologyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState<string>("All");

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
    axios.post<MorphologyResult>(`${base}/morphology/from-prediction/${id}`)
      .then(({ data }) => setResult(data))
      .catch(() => setError("Morphology classification failed."))
      .finally(() => setLoading(false));
  }, [id]);

  if (!id) return <p className="text-muted-foreground">No prediction ID provided. Go to Detection Results and click Analyse Morphology.</p>;
  if (loading) return <div className="flex items-center gap-2 text-muted-foreground"><Loader2 className="w-4 h-4 animate-spin" /> Running ResNet18 morphology classification...</div>;
  if (error) return <p className="text-red-400">{error}</p>;
  if (!result) return null;

  const style = SEVERITY_STYLES[result.severity] ?? SEVERITY_STYLES.Normal;
  const abnormalCells = result.per_cell_predictions.filter(c => c.label !== "Normal");
  const filtered = filter === "All" ? result.per_cell_predictions : result.per_cell_predictions.filter(c => c.label === filter);

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <FlaskConical className="w-6 h-6" /> RBC Morphology — Agent 3
        </h1>
        <p className="text-muted-foreground text-sm mt-1">ResNet18 · Classes: Normal · Sickle · Crescent · Elongated · For research use only</p>
      </div>

      {/* Severity Banner */}
      <div className={`rounded-lg border p-4 flex items-center gap-4 ${style.color}`}>
        {style.icon}
        <div>
          <p className="font-bold text-lg">{result.severity} — {result.abnormal_pct}% Abnormal RBC</p>
          <p className="text-sm mt-0.5 opacity-90">{result.recommendation}</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Total RBC" value={result.total_rbc} />
        <StatCard title="Normal" value={result.normal_count} color="green" />
        <StatCard title="Abnormal" value={result.abnormal_count} color="red" />
        <StatCard title="Abnormal %" value={`${result.abnormal_pct}%`} color={result.abnormal_pct > 30 ? "red" : result.abnormal_pct > 0 ? "default" : "green"} />
      </div>

      {/* Class breakdown */}
      <div className="rounded-lg border border-border bg-card p-4">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Class Breakdown</p>
        <div className="grid grid-cols-4 gap-3 text-center">
          {Object.entries(result.class_counts).map(([cls, count]) => (
            <div key={cls} className="rounded-lg border border-border p-3">
              <p className={`text-2xl font-bold ${LABEL_COLORS[cls]}`}>{count}</p>
              <p className="text-xs text-muted-foreground mt-1">{cls}</p>
              <p className="text-xs text-muted-foreground">{result.total_rbc > 0 ? ((count / result.total_rbc) * 100).toFixed(1) : 0}%</p>
            </div>
          ))}
        </div>
      </div>

      {/* Abnormal cell images */}
      {abnormalCells.length > 0 && (
        <div className="rounded-lg border border-border bg-card p-4">
          <p className="text-sm font-medium mb-3 text-red-400">Abnormal Cells ({abnormalCells.length})</p>
          <div className="flex flex-wrap gap-2">
            {abnormalCells.map((c) => (
              <div key={c.cell_index} className="rounded border border-red-500/30 bg-red-500/5 p-1 text-center">
                {c.crop_url && (
                  <img src={`http://localhost:8000${c.crop_url}`} alt={c.label}
                    className="w-16 h-16 object-cover rounded" />
                )}
                <p className={`text-xs mt-1 font-medium ${LABEL_COLORS[c.label]}`}>{c.label}</p>
                <p className="text-xs text-muted-foreground">{(c.confidence * 100).toFixed(1)}%</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Per-cell table */}
      <div className="rounded-lg border border-border overflow-hidden">
        <div className="px-4 py-3 bg-muted/30 flex items-center justify-between">
          <p className="text-sm font-medium">All Cells ({result.per_cell_predictions.length})</p>
          <div className="flex gap-1">
            {["All", "Normal", "Sickle", "Crescent", "Elongated"].map((f) => (
              <button key={f} onClick={() => setFilter(f)}
                className={`px-2 py-1 rounded text-xs transition-colors ${filter === f ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-accent"}`}>
                {f}
              </button>
            ))}
          </div>
        </div>
        <div className="overflow-auto max-h-72">
          <table className="w-full text-sm">
            <thead className="bg-muted/20 sticky top-0">
              <tr>
                {["Cell #", "Label", "Confidence", "Normal", "Sickle", "Crescent", "Elongated"].map((h) => (
                  <th key={h} className="px-4 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filtered.map((c) => (
                <tr key={c.cell_index} className="hover:bg-accent/20">
                  <td className="px-4 py-2 font-mono text-xs">{c.cell_index + 1}</td>
                  <td className="px-4 py-2"><span className={`font-medium ${LABEL_COLORS[c.label]}`}>{c.label}</span></td>
                  <td className="px-4 py-2">{(c.confidence * 100).toFixed(1)}%</td>
                  {["Normal", "Sickle", "Crescent", "Elongated"].map((cls) => (
                    <td key={cls} className="px-4 py-2 text-xs text-muted-foreground">
                      {((c.probabilities[cls] ?? 0) * 100).toFixed(1)}%
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-xs text-muted-foreground">⚠️ For research use only. Not a medical diagnostic tool.</p>
    </div>
  );
}

export default function MorphologyPage() {
  return (
    <Suspense fallback={<p className="text-muted-foreground">Loading...</p>}>
      <MorphologyContent />
    </Suspense>
  );
}

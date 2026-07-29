"use client";
import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import axios from "axios";
import { Loader2, Microscope } from "lucide-react";
import { StatCard } from "@/components/stat-card";

interface CellPrediction {
  cell_index: number;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  crop_url?: string;
}

interface DifferentialEntry {
  count: number;
  pct: number;
  normal_range: string;
  status: "Normal" | "High" | "Low" | "Abnormal";
}

interface WBCResult {
  total_wbc: number;
  class_counts: Record<string, number>;
  class_pct: Record<string, number>;
  differential: Record<string, DifferentialEntry>;
  dominant_type: string;
  per_cell_predictions: CellPrediction[];
}

const STATUS_COLORS: Record<string, string> = {
  Normal:   "text-green-400",
  High:     "text-red-400",
  Low:      "text-yellow-400",
  Abnormal: "text-orange-400",
};

const CLASS_COLORS: Record<string, string> = {
  basophil:    "text-purple-400",
  eosinophil:  "text-pink-400",
  erythroblast:"text-red-400",
  ig:          "text-orange-400",
  lymphocyte:  "text-blue-400",
  monocyte:    "text-yellow-400",
  neutrophil:  "text-green-400",
  platelet:    "text-cyan-400",
};

function WBCContent() {
  const params = useSearchParams();
  const id = params.get("id");
  const [result, setResult] = useState<WBCResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("All");

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
    axios.post<WBCResult>(`${base}/wbc/from-prediction/${id}`)
      .then(({ data }) => setResult(data))
      .catch(() => setError("WBC classification failed."))
      .finally(() => setLoading(false));
  }, [id]);

  if (!id) return <p className="text-muted-foreground">No prediction ID provided. Go to Detection Results and click Classify WBC.</p>;
  if (loading) return <div className="flex items-center gap-2 text-muted-foreground"><Loader2 className="w-4 h-4 animate-spin" /> Running SigLIP WBC classification...</div>;
  if (error) return <p className="text-red-400">{error}</p>;
  if (!result) return null;

  const classes = Object.keys(result.class_counts);
  const filtered = filter === "All"
    ? result.per_cell_predictions
    : result.per_cell_predictions.filter(c => c.label === filter);

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Microscope className="w-6 h-6" /> WBC Sub-type Classifier — Agent 4
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          SigLIP · 8 classes · basophil · eosinophil · erythroblast · ig · lymphocyte · monocyte · neutrophil · platelet · For research use only
        </p>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        <StatCard title="Total WBC" value={result.total_wbc} icon={<Microscope className="w-4 h-4" />} />
        <StatCard title="Dominant Type" value={result.dominant_type} />
        <StatCard title="Unique Types" value={classes.filter(c => result.class_counts[c] > 0).length} color="blue" />
      </div>

      {/* Class breakdown */}
      <div className="rounded-lg border border-border bg-card p-4">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Class Distribution</p>
        <div className="grid grid-cols-4 gap-3 text-center">
          {classes.map((cls) => (
            <div key={cls} className="rounded-lg border border-border p-3">
              <p className={`text-2xl font-bold ${CLASS_COLORS[cls] ?? "text-foreground"}`}>{result.class_counts[cls]}</p>
              <p className="text-xs font-medium mt-1 capitalize">{cls}</p>
              <p className="text-xs text-muted-foreground">{result.class_pct[cls]}%</p>
            </div>
          ))}
        </div>
      </div>

      {/* Differential table */}
      <div className="rounded-lg border border-border overflow-hidden">
        <div className="px-4 py-3 bg-muted/30 text-sm font-medium">WBC Differential</div>
        <table className="w-full text-sm">
          <thead className="bg-muted/20">
            <tr>
              {["Cell Type", "Count", "%", "Normal Range", "Status"].map(h => (
                <th key={h} className="px-4 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {Object.entries(result.differential).map(([cls, d]) => (
              <tr key={cls} className="hover:bg-accent/20">
                <td className={`px-4 py-2 font-medium capitalize ${CLASS_COLORS[cls] ?? ""}`}>{cls}</td>
                <td className="px-4 py-2">{d.count}</td>
                <td className="px-4 py-2">{d.pct}%</td>
                <td className="px-4 py-2 text-xs text-muted-foreground">{d.normal_range}</td>
                <td className={`px-4 py-2 font-medium ${STATUS_COLORS[d.status]}`}>{d.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Cell images by type */}
      {classes.filter(c => result.class_counts[c] > 0).map(cls => {
        const cells = result.per_cell_predictions.filter(c => c.label === cls && c.crop_url);
        if (!cells.length) return null;
        return (
          <div key={cls} className="rounded-lg border border-border bg-card p-4">
            <p className={`text-sm font-medium mb-3 capitalize ${CLASS_COLORS[cls]}`}>{cls} ({cells.length})</p>
            <div className="flex flex-wrap gap-2">
              {cells.map(c => (
                <div key={c.cell_index} className="rounded border border-border p-1 text-center">
                  <img src={`http://localhost:8000${c.crop_url}`} alt={c.label}
                    className="w-16 h-16 object-cover rounded" />
                  <p className="text-xs text-muted-foreground mt-1">{(c.confidence * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {/* Per-cell table */}
      <div className="rounded-lg border border-border overflow-hidden">
        <div className="px-4 py-3 bg-muted/30 flex items-center justify-between">
          <p className="text-sm font-medium">All WBC Predictions ({result.per_cell_predictions.length})</p>
          <div className="flex flex-wrap gap-1">
            {["All", ...classes].map(f => (
              <button key={f} onClick={() => setFilter(f)}
                className={`px-2 py-1 rounded text-xs capitalize transition-colors ${filter === f ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-accent"}`}>
                {f}
              </button>
            ))}
          </div>
        </div>
        <div className="overflow-auto max-h-72">
          <table className="w-full text-sm">
            <thead className="bg-muted/20 sticky top-0">
              <tr>
                {["Cell #", "Label", "Confidence"].map(h => (
                  <th key={h} className="px-4 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">{h}</th>
                ))}
                {classes.map(c => (
                  <th key={c} className="px-4 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider capitalize">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filtered.map(c => (
                <tr key={c.cell_index} className="hover:bg-accent/20">
                  <td className="px-4 py-2 font-mono text-xs">{c.cell_index + 1}</td>
                  <td className={`px-4 py-2 font-medium capitalize ${CLASS_COLORS[c.label] ?? ""}`}>{c.label}</td>
                  <td className="px-4 py-2">{(c.confidence * 100).toFixed(1)}%</td>
                  {classes.map(cls => (
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

      <p className="text-xs text-muted-foreground">⚠️ For research use only. Not a medical diagnostic tool. Clinical review required.</p>
    </div>
  );
}

export default function WBCPage() {
  return (
    <Suspense fallback={<p className="text-muted-foreground">Loading...</p>}>
      <WBCContent />
    </Suspense>
  );
}

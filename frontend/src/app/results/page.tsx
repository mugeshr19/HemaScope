"use client";
import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import axios from "axios";
import { ImageViewer } from "@/components/image-viewer";
import { getAnnotatedImageUrl, getDownloadUrl } from "@/lib/api";
import type { PredictionResult } from "@/types";
import {
  Download, Microscope, Activity, Loader2,
  ShieldAlert, ShieldCheck, ShieldQuestion, FlaskConical, Dna,
} from "lucide-react";

// ── Risk styles for Agent 2 ───────────────────────────────────────────────────
const RISK_STYLES: Record<string, { color: string; icon: React.ReactNode }> = {
  Negative: { color: "text-green-400 border-green-500/30 bg-green-500/5",   icon: <ShieldCheck className="w-5 h-5 text-green-400" /> },
  Low:      { color: "text-yellow-400 border-yellow-500/30 bg-yellow-500/5", icon: <ShieldQuestion className="w-5 h-5 text-yellow-400" /> },
  Moderate: { color: "text-orange-400 border-orange-500/30 bg-orange-500/5", icon: <ShieldAlert className="w-5 h-5 text-orange-400" /> },
  High:     { color: "text-red-400 border-red-500/30 bg-red-500/5",          icon: <ShieldAlert className="w-5 h-5 text-red-400" /> },
};

// ── Severity styles for Agent 3 ───────────────────────────────────────────────
const SEVERITY_STYLES: Record<string, { color: string; icon: React.ReactNode }> = {
  Normal:   { color: "text-green-400 border-green-500/30 bg-green-500/5",   icon: <ShieldCheck className="w-5 h-5 text-green-400" /> },
  Mild:     { color: "text-yellow-400 border-yellow-500/30 bg-yellow-500/5", icon: <ShieldQuestion className="w-5 h-5 text-yellow-400" /> },
  Moderate: { color: "text-orange-400 border-orange-500/30 bg-orange-500/5", icon: <ShieldAlert className="w-5 h-5 text-orange-400" /> },
  Severe:   { color: "text-red-400 border-red-500/30 bg-red-500/5",          icon: <ShieldAlert className="w-5 h-5 text-red-400" /> },
};

const MORPH_COLORS: Record<string, string> = {
  Normal: "text-green-400", Sickle: "text-red-400",
  Crescent: "text-orange-400", Elongated: "text-yellow-400",
};

const WBC_COLORS: Record<string, string> = {
  basophil: "text-purple-400", eosinophil: "text-pink-400",
  erythroblast: "text-red-400", ig: "text-orange-400",
  lymphocyte: "text-blue-400", monocyte: "text-yellow-400",
  neutrophil: "text-green-400", platelet: "text-cyan-400",
};

const STATUS_COLORS: Record<string, string> = {
  Normal: "text-green-400", High: "text-red-400",
  Low: "text-yellow-400", Abnormal: "text-orange-400",
};

// ── Agent panel wrapper ───────────────────────────────────────────────────────
function AgentPanel({ title, subtitle, icon, onRun, loading, ran, children }: {
  title: string; subtitle: string; icon: React.ReactNode;
  onRun: () => void; loading: boolean; ran: boolean; children?: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-border bg-card p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="font-semibold flex items-center gap-2">{icon}{title}</p>
          <p className="text-xs text-muted-foreground mt-0.5">{subtitle}</p>
        </div>
        {!ran && (
          <button onClick={onRun} disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium disabled:opacity-50 hover:bg-primary/90 transition-colors">
            {loading ? <><Loader2 className="w-4 h-4 animate-spin" />Running...</> : icon}
            {!loading && title.split("—")[0].trim()}
          </button>
        )}
      </div>
      {children}
    </div>
  );
}

function ResultsContent() {
  const params = useSearchParams();
  const id = params.get("id");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState("");
  const detections = result?.detections ?? [];

  // Agent 2 — Malaria
  const [malaria, setMalaria] = useState<Record<string, unknown> | null>(null);
  const [malariaLoading, setMalariaLoading] = useState(false);

  // Agent 3 — Morphology
  const [morph, setMorph] = useState<Record<string, unknown> | null>(null);
  const [morphLoading, setMorphLoading] = useState(false);

  // Agent 4 — WBC
  const [wbc, setWbc] = useState<Record<string, unknown> | null>(null);
  const [wbcLoading, setWbcLoading] = useState(false);

  const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

  useEffect(() => {
    if (!id) return;
    const cached = sessionStorage.getItem(`result_${id}`);
    if (cached) { setResult(JSON.parse(cached)); return; }
    axios.get<PredictionResult[]>(`${base}/history`)
      .then(({ data }) => {
        const found = data.find((r) => r.prediction_id === id);
        if (found) setResult(found);
        else setError("Result not found.");
      })
      .catch(() => setError("Failed to load result."));
  }, [id]);

  const runMalaria = async () => {
    setMalariaLoading(true);
    try { const { data } = await axios.post(`${base}/malaria/from-prediction/${id}`); setMalaria(data); }
    catch { setMalaria({ error: "Malaria screening failed." }); }
    finally { setMalariaLoading(false); }
  };

  const runMorph = async () => {
    setMorphLoading(true);
    try { const { data } = await axios.post(`${base}/morphology/from-prediction/${id}`); setMorph(data); }
    catch { setMorph({ error: "Morphology classification failed." }); }
    finally { setMorphLoading(false); }
  };

  const runWbc = async () => {
    setWbcLoading(true);
    try { const { data } = await axios.post(`${base}/wbc/from-prediction/${id}`); setWbc(data); }
    catch { setWbc({ error: "WBC classification failed." }); }
    finally { setWbcLoading(false); }
  };

  if (!id) return <p className="text-muted-foreground">No prediction ID provided.</p>;
  if (error) return <p className="text-red-400">{error}</p>;
  if (!result) return <p className="text-muted-foreground">Loading...</p>;

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2"><Microscope className="w-6 h-6" /> Detection Results</h1>
          <p className="text-muted-foreground text-sm mt-1">{result.image_name}</p>
        </div>
        <div className="flex gap-2">
          <a href={getDownloadUrl(id, "json")} className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-border text-sm hover:bg-accent transition-colors">
            <Download className="w-4 h-4" /> JSON
          </a>
          <a href={getDownloadUrl(id, "csv")} className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-border text-sm hover:bg-accent transition-colors">
            <Download className="w-4 h-4" /> CSV
          </a>
        </div>
      </div>

      {/* Counts */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: "Total Cells", value: result.total_cells, color: "text-foreground" },
          { label: "RBC", value: result.rbc, color: "text-red-400" },
          { label: "WBC", value: result.wbc, color: "text-green-400" },
          { label: "Platelets", value: result.platelet, color: "text-blue-400" },
        ].map(({ label, value, color }) => (
          <div key={label} className="rounded-lg border border-border bg-card p-4 text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wider">{label}</p>
            <p className={`text-3xl font-bold mt-1 ${color}`}>{value}</p>
          </div>
        ))}
      </div>

      {/* Annotated image */}
      <div className="rounded-lg border border-border bg-card p-4">
        <p className="text-sm font-medium mb-3">Annotated Image</p>
        <ImageViewer src={getAnnotatedImageUrl(id)} alt="Annotated blood smear" />
      </div>

      {/* Detections table */}
      <div className="rounded-lg border border-border overflow-hidden">
        <div className="px-4 py-3 bg-muted/30 text-sm font-medium">Detections ({detections.length})</div>
        <div className="overflow-auto max-h-72">
          <table className="w-full text-sm">
            <thead className="bg-muted/20 sticky top-0">
              <tr>{["Cell ID", "Type", "Confidence", "BBox"].map(h => (
                <th key={h} className="px-4 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">{h}</th>
              ))}</tr>
            </thead>
            <tbody className="divide-y divide-border">
              {detections.map(d => (
                <tr key={d.cell_id} className="hover:bg-accent/20">
                  <td className="px-4 py-2 font-mono text-xs">{d.cell_id}</td>
                  <td className="px-4 py-2">
                    <span className={`font-medium ${d.class === "RBC" ? "text-red-400" : d.class === "WBC" ? "text-green-400" : "text-blue-400"}`}>{d.class}</span>
                  </td>
                  <td className="px-4 py-2">{(d.confidence * 100).toFixed(1)}%</td>
                  <td className="px-4 py-2 font-mono text-xs text-muted-foreground">[{d.bbox.map(v => Math.round(v)).join(", ")}]</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-xs text-muted-foreground">Inference time: {result.inference_time}s · ID: {id}</p>

      {/* ── Agent 2: Malaria ─────────────────────────────────────────────────── */}
      <AgentPanel
        title="Malaria Screening — Agent 2"
        subtitle={`Run DenseNet-121 on the ${result.rbc} RBC crops`}
        icon={<Activity className="w-4 h-4" />}
        onRun={runMalaria} loading={malariaLoading} ran={!!malaria}
      >
        {malaria && !("error" in malaria) && (() => {
          const r = malaria as any;
          const style = RISK_STYLES[r.risk_level] ?? RISK_STYLES.Negative;
          const infected = (r.per_cell_predictions ?? []).filter((c: any) => c.label === "Parasitized");
          return (
            <div className="space-y-3">
              <div className={`rounded-lg border p-4 flex items-center gap-4 ${style.color}`}>
                {style.icon}
                <div>
                  <p className="font-bold">{r.risk_level} Risk · Density: {r.parasite_density_pct}% · Infected: {r.infected_rbc}/{r.total_rbc} RBC</p>
                  <p className="text-sm mt-0.5 opacity-90">{r.recommendation}</p>
                </div>
              </div>
              {infected.length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-2 text-red-400">Infected Cells ({infected.length})</p>
                  <div className="flex flex-wrap gap-2">
                    {infected.map((c: any) => (
                      <div key={c.cell_index} className="rounded border border-red-500/40 bg-red-500/5 p-1 text-center">
                        <img src={`http://localhost:8000${c.crop_url}`} className="w-16 h-16 object-cover rounded" />
                        <p className="text-xs text-red-400 mt-1">{(c.p_infected * 100).toFixed(1)}%</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })()}
        {"error" in (malaria ?? {}) && <p className="text-red-400 text-sm">{String((malaria as any).error)}</p>}
      </AgentPanel>

      {/* ── Agent 3: Morphology ──────────────────────────────────────────────── */}
      <AgentPanel
        title="RBC Morphology — Agent 3"
        subtitle={`Classify each RBC as Normal / Sickle / Crescent / Elongated`}
        icon={<FlaskConical className="w-4 h-4" />}
        onRun={runMorph} loading={morphLoading} ran={!!morph}
      >
        {morph && !("error" in morph) && (() => {
          const r = morph as any;
          const style = SEVERITY_STYLES[r.severity] ?? SEVERITY_STYLES.Normal;
          const abnormal = (r.per_cell_predictions ?? []).filter((c: any) => c.label !== "Normal");
          return (
            <div className="space-y-3">
              <div className={`rounded-lg border p-4 flex items-center gap-4 ${style.color}`}>
                {style.icon}
                <div>
                  <p className="font-bold">{r.severity} · {r.abnormal_pct}% Abnormal · {r.abnormal_count}/{r.total_rbc} RBC</p>
                  <p className="text-sm mt-0.5 opacity-90">{r.recommendation}</p>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-2 text-center text-sm">
                {Object.entries(r.class_counts as Record<string, number>).map(([cls, cnt]) => (
                  <div key={cls} className="rounded border border-border p-2">
                    <p className={`text-xl font-bold ${MORPH_COLORS[cls] ?? ""}`}>{cnt}</p>
                    <p className="text-xs text-muted-foreground">{cls}</p>
                  </div>
                ))}
              </div>
              {abnormal.length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-2 text-red-400">Abnormal Cells ({abnormal.length})</p>
                  <div className="flex flex-wrap gap-2">
                    {abnormal.map((c: any) => (
                      <div key={c.cell_index} className="rounded border border-red-500/30 bg-red-500/5 p-1 text-center">
                        {c.crop_url && <img src={`http://localhost:8000${c.crop_url}`} className="w-16 h-16 object-cover rounded" />}
                        <p className={`text-xs mt-1 font-medium ${MORPH_COLORS[c.label]}`}>{c.label}</p>
                        <p className="text-xs text-muted-foreground">{(c.confidence * 100).toFixed(1)}%</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })()}
        {"error" in (morph ?? {}) && <p className="text-red-400 text-sm">{String((morph as any).error)}</p>}
      </AgentPanel>

      {/* ── Agent 4: WBC ─────────────────────────────────────────────────────── */}
      <AgentPanel
        title="WBC Sub-type Classifier — Agent 4"
        subtitle={`Classify each WBC into 8 sub-types using SigLIP`}
        icon={<Dna className="w-4 h-4" />}
        onRun={runWbc} loading={wbcLoading} ran={!!wbc}
      >
        {wbc && !("error" in wbc) && (() => {
          const r = wbc as any;
          const classes: string[] = Object.keys(r.class_counts ?? {});
          return (
            <div className="space-y-3">
              <div className="rounded-lg border border-border bg-muted/10 p-3 flex items-center gap-4">
                <Dna className="w-5 h-5 text-primary" />
                <div>
                  <p className="font-bold">Dominant: <span className={WBC_COLORS[r.dominant_type] ?? ""}>{r.dominant_type}</span> · Total WBC: {r.total_wbc}</p>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-2 text-center text-sm">
                {classes.map(cls => (
                  <div key={cls} className="rounded border border-border p-2">
                    <p className={`text-xl font-bold ${WBC_COLORS[cls] ?? ""}`}>{r.class_counts[cls]}</p>
                    <p className="text-xs text-muted-foreground capitalize">{cls}</p>
                    <p className="text-xs text-muted-foreground">{r.class_pct[cls]}%</p>
                  </div>
                ))}
              </div>
              {/* Differential */}
              <div className="rounded-lg border border-border overflow-hidden">
                <div className="px-3 py-2 bg-muted/30 text-xs font-medium uppercase tracking-wider text-muted-foreground">Differential</div>
                <table className="w-full text-xs">
                  <thead className="bg-muted/20">
                    <tr>{["Type", "Count", "%", "Normal Range", "Status"].map(h => (
                      <th key={h} className="px-3 py-1.5 text-left font-medium text-muted-foreground">{h}</th>
                    ))}</tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {Object.entries(r.differential as Record<string, any>).map(([cls, d]: [string, any]) => (
                      <tr key={cls} className="hover:bg-accent/20">
                        <td className={`px-3 py-1.5 font-medium capitalize ${WBC_COLORS[cls] ?? ""}`}>{cls}</td>
                        <td className="px-3 py-1.5">{d.count}</td>
                        <td className="px-3 py-1.5">{d.pct}%</td>
                        <td className="px-3 py-1.5 text-muted-foreground">{d.normal_range}</td>
                        <td className={`px-3 py-1.5 font-medium ${STATUS_COLORS[d.status] ?? ""}`}>{d.status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {/* WBC crop images grouped by type */}
              {classes.filter(c => r.class_counts[c] > 0).map((cls: string) => {
                const cells = (r.per_cell_predictions ?? []).filter((c: any) => c.label === cls && c.crop_url);
                if (!cells.length) return null;
                return (
                  <div key={cls}>
                    <p className={`text-sm font-medium mb-2 capitalize ${WBC_COLORS[cls] ?? ""}`}>{cls} ({cells.length})</p>
                    <div className="flex flex-wrap gap-2">
                      {cells.map((c: any) => (
                        <div key={c.cell_index} className="rounded border border-border p-1 text-center">
                          <img src={`http://localhost:8000${c.crop_url}`} className="w-16 h-16 object-cover rounded" />
                          <p className="text-xs text-muted-foreground mt-1">{(c.confidence * 100).toFixed(1)}%</p>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          );
        })()}
        {"error" in (wbc ?? {}) && <p className="text-red-400 text-sm">{String((wbc as any).error)}</p>}
      </AgentPanel>

    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={<p className="text-muted-foreground">Loading...</p>}>
      <ResultsContent />
    </Suspense>
  );
}

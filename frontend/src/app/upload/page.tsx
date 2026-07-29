"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Dropzone } from "@/components/dropzone";
import { predictImage } from "@/lib/api";
import { Loader2, Microscope } from "lucide-react";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return null;
  const router = useRouter();

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    try {
      const result = await predictImage(file);
      sessionStorage.setItem(`result_${result.prediction_id}`, JSON.stringify(result));
      toast.success(`Detected ${result.total_cells} cells in ${result.inference_time}s`);
      router.push(`/results?id=${result.prediction_id}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Prediction failed";
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Upload Image</h1>
        <p className="text-muted-foreground text-sm mt-1">Upload a blood smear image to run YOLOv11 detection</p>
      </div>

      <div className="rounded-lg border border-border bg-card p-6 space-y-4">
        <Dropzone onFile={setFile} disabled={loading} />

        <button
          onClick={handleSubmit}
          disabled={!file || loading}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-md bg-primary text-primary-foreground font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-primary/90 transition-colors"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running Detection...
            </>
          ) : (
            <>
              <Microscope className="w-4 h-4" />
              Detect Blood Cells
            </>
          )}
        </button>
      </div>

      <div className="rounded-lg border border-border bg-card p-4">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Detection Pipeline</p>
        <ol className="space-y-1.5 text-sm text-muted-foreground">
          {["Upload Image", "Run YOLOv11", "Detect & Classify Cells", "Draw Bounding Boxes",
            "Crop Every Cell", "Calculate Counts", "Export JSON & CSV"].map((step, i) => (
            <li key={i} className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center shrink-0">
                {i + 1}
              </span>
              {step}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

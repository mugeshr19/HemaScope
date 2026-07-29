"use client";
import { useState } from "react";
import { toast } from "sonner";

const DEFAULT = {
  apiUrl: "http://localhost:8000/api/v1",
  confidence: "0.25",
  iou: "0.45",
  imageSize: "640",
  maxDetections: "1000",
};

export default function SettingsPage() {
  const [form, setForm] = useState(DEFAULT);

  const save = () => {
    localStorage.setItem("bcd_settings", JSON.stringify(form));
    toast.success("Settings saved");
  };

  const field = (label: string, key: keyof typeof form, hint?: string) => (
    <div key={key}>
      <label className="block text-sm font-medium mb-1">{label}</label>
      {hint && <p className="text-xs text-muted-foreground mb-1">{hint}</p>}
      <input
        value={form[key]}
        onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
        className="w-full px-3 py-2 rounded-md border border-border bg-card text-sm focus:outline-none focus:ring-1 focus:ring-ring"
      />
    </div>
  );

  return (
    <div className="max-w-xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-muted-foreground text-sm mt-1">Configure inference parameters</p>
      </div>

      <div className="rounded-lg border border-border bg-card p-5 space-y-4">
        <p className="text-sm font-semibold">API Configuration</p>
        {field("API URL", "apiUrl")}
      </div>

      <div className="rounded-lg border border-border bg-card p-5 space-y-4">
        <p className="text-sm font-semibold">Inference Parameters</p>
        {field("Confidence Threshold", "confidence", "Minimum confidence score (0–1)")}
        {field("IoU Threshold", "iou", "Non-maximum suppression IoU threshold")}
        {field("Image Size", "imageSize", "Input image size for YOLOv11")}
        {field("Max Detections", "maxDetections", "Maximum number of detections per image")}
      </div>

      <button
        onClick={save}
        className="px-5 py-2 rounded-md bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
      >
        Save Settings
      </button>
    </div>
  );
}

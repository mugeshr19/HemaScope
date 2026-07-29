"use client";
import { useEffect, useState } from "react";
import { getHistory } from "@/lib/api";
import type { PredictionSummary } from "@/types";
import { formatDate } from "@/lib/utils";
import { useRouter } from "next/navigation";
import { Search, ChevronRight } from "lucide-react";

export default function HistoryPage() {
  const [records, setRecords] = useState<PredictionSummary[]>([]);
  const [query, setQuery] = useState("");
  const router = useRouter();

  useEffect(() => {
    getHistory(0, 100).then(setRecords).catch(console.error);
  }, []);

  const filtered = records.filter((r) =>
    r.image_name.toLowerCase().includes(query.toLowerCase()) ||
    r.prediction_id.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Prediction History</h1>
        <p className="text-muted-foreground text-sm mt-1">{records.length} predictions stored</p>
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by image name or ID..."
          className="w-full pl-9 pr-4 py-2 rounded-md border border-border bg-card text-sm focus:outline-none focus:ring-1 focus:ring-ring"
        />
      </div>

      <div className="rounded-lg border border-border overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-muted/30">
            <tr>
              {["Image", "Date", "Total", "RBC", "WBC", "Platelet", "Time", ""].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-muted-foreground">
                  No predictions found
                </td>
              </tr>
            ) : (
              filtered.map((r) => (
                <tr
                  key={r.prediction_id}
                  className="hover:bg-accent/30 cursor-pointer transition-colors"
                  onClick={() => router.push(`/results?id=${r.prediction_id}`)}
                >
                  <td className="px-4 py-3 font-medium truncate max-w-[160px]">{r.image_name}</td>
                  <td className="px-4 py-3 text-muted-foreground">{formatDate(r.timestamp)}</td>
                  <td className="px-4 py-3 font-semibold">{r.total_cells}</td>
                  <td className="px-4 py-3 text-red-400">{r.rbc}</td>
                  <td className="px-4 py-3 text-green-400">{r.wbc}</td>
                  <td className="px-4 py-3 text-blue-400">{r.platelet}</td>
                  <td className="px-4 py-3 text-muted-foreground">{r.inference_time}s</td>
                  <td className="px-4 py-3">
                    <ChevronRight className="w-4 h-4 text-muted-foreground" />
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

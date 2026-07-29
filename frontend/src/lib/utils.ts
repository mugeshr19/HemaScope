import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const CLASS_COLORS: Record<string, string> = {
  RBC: "#ef4444",
  WBC: "#22c55e",
  Platelet: "#3b82f6",
};

export const CLASS_BG: Record<string, string> = {
  RBC: "bg-red-500/10 text-red-400 border-red-500/20",
  WBC: "bg-green-500/10 text-green-400 border-green-500/20",
  Platelet: "bg-blue-500/10 text-blue-400 border-blue-500/20",
};

export const formatDate = (iso: string) =>
  new Date(iso).toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });

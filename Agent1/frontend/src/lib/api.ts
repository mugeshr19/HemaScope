import axios from "axios";
import type { PredictionResult, PredictionSummary, Metrics, HealthStatus } from "@/types";

const api = axios.create({ baseURL: process.env.NEXT_PUBLIC_API_URL });

export const predictImage = async (file: File): Promise<PredictionResult> => {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<PredictionResult>("/predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const getHistory = async (skip = 0, limit = 50): Promise<PredictionSummary[]> => {
  const { data } = await api.get<PredictionSummary[]>("/history", { params: { skip, limit } });
  return data;
};

export const getMetrics = async (): Promise<Metrics> => {
  const { data } = await api.get<Metrics>("/metrics");
  return data;
};

export const getHealth = async (): Promise<HealthStatus> => {
  const { data } = await api.get<HealthStatus>("/health");
  return data;
};

export const getAnnotatedImageUrl = (predictionId: string) =>
  `${process.env.NEXT_PUBLIC_API_URL}/results/${predictionId}/annotated`;

export const getDownloadUrl = (predictionId: string, type: "json" | "csv" | "crops") =>
  `${process.env.NEXT_PUBLIC_API_URL}/results/${predictionId}/download/${type}`;

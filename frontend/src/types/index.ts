export interface Detection {
  cell_id: string;
  class: "RBC" | "WBC" | "Platelet";
  confidence: number;
  bbox: [number, number, number, number];
  crop_path: string;
}

export interface PredictionResult {
  prediction_id: string;
  image_name: string;
  total_cells: number;
  rbc: number;
  wbc: number;
  platelet: number;
  inference_time: number;
  timestamp: string;
  annotated_image_url: string | null;
  detections: Detection[];
}

export interface PredictionSummary {
  prediction_id: string;
  image_name: string;
  timestamp: string;
  total_cells: number;
  rbc: number;
  wbc: number;
  platelet: number;
  inference_time: number;
}

export interface Metrics {
  total_predictions: number;
  total_cells_detected: number;
  avg_inference_time: number;
  avg_rbc_per_image: number;
  avg_wbc_per_image: number;
  avg_platelet_per_image: number;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  database_connected: boolean;
  version: string;
}

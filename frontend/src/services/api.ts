import { 
  DatasetMetadata, 
  ClassifyRequest, 
  ClassifyResponse, 
  CompareRequest, 
  CompareResponse, 
  ExportRequest, 
  ExportResponse,
  DatasetPreviewResponse
} from "../types";

const API_BASE_URL = "/api/v1";

class ApiService {
  /**
   * Helper to handle response status and JSON conversions
   */
  private async request<T>(path: string, options: RequestInit): Promise<T> {
    const url = `${API_BASE_URL}${path}`;
    const response = await fetch(url, options);
    
    if (!response.ok) {
      let errorDetail = "API Request failed";
      try {
        const errorJson = await response.json();
        errorDetail = errorJson.detail || errorDetail;
      } catch {
        // Fallback if not JSON
      }
      throw new Error(errorDetail);
    }
    
    return response.json() as Promise<T>;
  }

  /**
   * Upload ARFF file
   */
  async uploadDataset(file: File): Promise<DatasetMetadata> {
    const formData = new FormData();
    formData.append("file", file);
    
    return this.request<DatasetMetadata>("/upload", {
      method: "POST",
      body: formData,
    });
  }

  /**
   * Fetch dataset metadata
   */
  async getDatasetMeta(datasetId: string): Promise<DatasetMetadata> {
    return this.request<DatasetMetadata>(`/dataset/info?dataset_id=${datasetId}`, {
      method: "GET",
    });
  }

  /**
   * Fetch dataset previews (raw and processed data)
   */
  async getDatasetPreview(datasetId: string, limit: number = 10): Promise<DatasetPreviewResponse> {
    return this.request<DatasetPreviewResponse>(`/dataset/preview?dataset_id=${datasetId}&limit=${limit}`, {
      method: "GET",
    });
  }

  /**
   * Train model with selected configuration
   */
  async trainClassifier(request: ClassifyRequest): Promise<ClassifyResponse> {
    return this.request<ClassifyResponse>("/classify/train", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
  }

  /**
   * Compare algorithms side by side
   */
  async compareClassifiers(request: CompareRequest): Promise<CompareResponse> {
    return this.request<CompareResponse>("/compare", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
  }

  /**
   * Export PDF or JSON report
   */
  async exportReport(request: ExportRequest): Promise<ExportResponse> {
    return this.request<ExportResponse>("/export", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
  }

  /**
   * Export Comparison CSV report file
   */
  async exportComparisonCSV(request: CompareRequest): Promise<Blob> {
    const url = `${API_BASE_URL}/compare/export-csv`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      throw new Error("Failed to export comparison CSV report");
    }
    
    return response.blob();
  }
}

export const apiService = new ApiService();

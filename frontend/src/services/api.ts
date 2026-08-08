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
   * Helper to handle response status and JSON conversions.
   * Throws an error that includes the HTTP status code, endpoint, and backend
   * error detail so users see a meaningful message instead of "API Request failed".
   */
  private async request<T>(path: string, options: RequestInit): Promise<T> {
    const url = `${API_BASE_URL}${path}`;
    let response: Response;

    try {
      response = await fetch(url, options);
    } catch (networkErr) {
      // Network-level failure (no connection, CORS preflight block, DNS failure, etc.)
      throw new Error(
        `Network error reaching ${url}: ${networkErr instanceof Error ? networkErr.message : String(networkErr)}`
      );
    }

    if (!response.ok) {
      let detail = `HTTP ${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        if (body?.detail) detail = `[${response.status}] ${body.detail}`;
        else detail = `[${response.status}] ${JSON.stringify(body)}`;
      } catch {
        // Response body wasn't JSON – use the status text
      }
      throw new Error(`${detail} (endpoint: ${url})`);
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

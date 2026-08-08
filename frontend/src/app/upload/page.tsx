"use client";

import React, { useState } from "react";
import { apiService } from "@/services/api";
import { useDataset } from "@/context/DatasetContext";
import { useToast } from "@/context/ToastContext";
import styles from "./page.module.css";
import commonStyles from "@/components/Common/Common.module.css";

export default function UploadPage() {
  const { setActiveDataset, datasetMetadata } = useDataset();
  const { showToast } = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith(".arff")) {
        setFile(selectedFile);
        setError(null);
        setSuccess(null);
      } else {
        setError("Invalid file format. Only .arff files are supported.");
        showToast("Invalid file format. Only .arff files are supported.", "error");
        setFile(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setSuccess(null);
    try {
      const res = await apiService.uploadDataset(file);
      
      // Fetch details of the dataset metadata to populate context fully
      const meta = await apiService.getDatasetMeta(res.dataset_id);
      
      // Update global context
      setActiveDataset(res.dataset_id, meta);
      setSuccess("Dataset uploaded, parsed, and validated successfully!");
      showToast("Dataset uploaded, parsed, and validated successfully!", "success");
      setFile(null);
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Failed to upload dataset.";
      setError(errMsg);
      showToast(errMsg, "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.pageTitle}>Upload Dataset</h1>
        <p className={styles.pageSubtitle}>Import your Attribute-Relation File Format (.arff) data files.</p>
      </div>

      <div className={styles.uploadGrid}>
        {/* Upload Form Card */}
        <div className={commonStyles.card}>
          <div className={styles.dropzone}>
            <span className={styles.dropzoneIcon}>📥</span>
            <p className={styles.dropzoneText}>Drag and drop your .arff file here</p>
            <p className={styles.dropzoneFormat}>Only standard ARFF files are supported</p>
            <input 
              type="file" 
              accept=".arff" 
              onChange={handleFileChange} 
              className={styles.fileInput}
              disabled={loading}
            />
          </div>

          {file && (
            <div className={styles.selectedFileCard}>
              <div className={styles.selectedFileInfo}>
                <span className={styles.fileIcon}>📄</span>
                <div className={styles.fileNameWrapper}>
                  <span className={styles.fileName}>{file.name}</span>
                  <span className={styles.fileSize}>{(file.size / 1024).toFixed(1)} KB</span>
                </div>
              </div>
              <button 
                onClick={handleUpload} 
                disabled={loading} 
                className={`${commonStyles.btn} ${commonStyles.btnPrimary} ${commonStyles.btnBlock}`}
              >
                {loading ? "Parsing Data..." : "Upload & Validate"}
              </button>
            </div>
          )}

          {loading && (
            <div className={commonStyles.spinnerWrapper}>
              <div className={commonStyles.spinner}></div>
              <span className={commonStyles.spinnerText}>Analyzing ARFF structure...</span>
            </div>
          )}

          {error && (
            <div className={`${commonStyles.alert} ${commonStyles.alertError}`} style={{ marginTop: "1.5rem" }}>
              <span>⚠️</span>
              <div>{error}</div>
            </div>
          )}

          {success && (
            <div className={`${commonStyles.alert} ${commonStyles.alertSuccess}`} style={{ marginTop: "1.5rem" }}>
              <span>✅</span>
              <div>{success}</div>
            </div>
          )}
        </div>

        {/* Info Card displaying current metadata */}
        {datasetMetadata && (
          <div className={commonStyles.card}>
            <h2 className={styles.sectionTitle}>Dataset Specifications</h2>
            <div className={styles.metadataGrid}>
              <div className={styles.metadataItem}>
                <span className={styles.metaLabel}>Relation Name</span>
                <span className={styles.metaValue}>{datasetMetadata.relation_name}</span>
              </div>
              <div className={styles.metadataItem}>
                <span className={styles.metaLabel}>Attributes Count</span>
                <span className={styles.metaValue}>{datasetMetadata.num_attributes}</span>
              </div>
              <div className={styles.metadataItem}>
                <span className={styles.metaLabel}>Instances Count</span>
                <span className={styles.metaValue}>{datasetMetadata.num_instances}</span>
              </div>
              <div className={styles.metadataItem}>
                <span className={styles.metaLabel}>Class Field</span>
                <span className={styles.metaValue} style={{ color: "var(--accent-primary)" }}>
                  {datasetMetadata.class_attribute || "None"}
                </span>
              </div>
            </div>

            <h3 className={styles.sectionTitle} style={{ fontSize: "1.1rem" }}>Schema Structure</h3>
            <div className={styles.tableWrapper}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Index</th>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Values</th>
                  </tr>
                </thead>
                <tbody>
                  {datasetMetadata.attributes.map((attr, idx) => (
                    <tr key={attr.name}>
                      <td>{idx + 1}</td>
                      <td className={styles.fontMono}>{attr.name}</td>
                      <td>
                        <span className={`${styles.typeBadge} ${styles[attr.type]}`}>
                          {attr.type}
                        </span>
                      </td>
                      <td className={styles.valuesCell} title={attr.values ? attr.values.join(", ") : ""}>
                        {attr.values ? attr.values.join(", ") : "N/A"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

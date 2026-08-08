"use client";

import React, { useEffect, useState } from "react";
import { useDataset } from "@/context/DatasetContext";
import { apiService } from "@/services/api";
import { DatasetPreviewResponse } from "@/types";
import styles from "./page.module.css";
import commonStyles from "@/components/Common/Common.module.css";
import Link from "next/link";

export default function PreviewPage() {
  const { activeDatasetId, datasetMetadata } = useDataset();
  const [preview, setPreview] = useState<DatasetPreviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"raw" | "processed">("raw");

  useEffect(() => {
    if (!activeDatasetId) return;
    
    const fetchPreview = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiService.getDatasetPreview(activeDatasetId, 15);
        setPreview(res);
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to load preview data.");
      } finally {
        setLoading(false);
      }
    };

    fetchPreview();
  }, [activeDatasetId]);

  if (!activeDatasetId || !datasetMetadata) {
    return (
      <div className={styles.container}>
        <div className={commonStyles.alert} style={{ backgroundColor: "rgba(239, 68, 68, 0.08)", border: "1px solid var(--color-error)" }}>
          <span>⚠️</span>
          <div>
            <h4 style={{ fontWeight: 600, marginBottom: "0.25rem" }}>Dataset not loaded</h4>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1rem" }}>
              You need to upload a dataset file first to inspect previews.
            </p>
            <Link href="/upload" className={`${commonStyles.btn} ${commonStyles.btnPrimary}`}>
              Upload Section &rarr;
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.pageTitle}>Dataset Preview</h1>
        <p className={styles.pageSubtitle}>
          Review instances in relation <strong>{datasetMetadata.relation_name}</strong>. Compare raw strings and numerical scales.
        </p>
      </div>

      {loading && (
        <div className={commonStyles.spinnerWrapper}>
          <div className={commonStyles.spinner}></div>
          <span className={commonStyles.spinnerText}>Loading preview rows from database...</span>
        </div>
      )}

      {error && (
        <div className={`${commonStyles.alert} ${commonStyles.alertError}`}>
          <span>⚠️</span>
          <div>{error}</div>
        </div>
      )}

      {preview && !loading && (
        <>
          <div className={styles.tabsContainer}>
            <button
              onClick={() => setActiveTab("raw")}
              className={`${styles.tab} ${activeTab === "raw" ? styles.activeTab : ""}`}
            >
              Raw Dataset
            </button>
            <button
              onClick={() => setActiveTab("processed")}
              className={`${styles.tab} ${activeTab === "processed" ? styles.activeTab : ""}`}
            >
              Preprocessed Dataset (Engine Outputs)
            </button>
          </div>

          <div className={styles.tableCard}>
            <h2 className={styles.tableTitle}>
              {activeTab === "raw" ? "Raw Instance Vectors" : "Cleaned, Encoded & Normalized Vectors"}
              <span className={styles.badge}>Showing 15 rows</span>
            </h2>

            <div className={styles.tableWrapper}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Index</th>
                    {preview.columns.map((col: string) => (
                      <th key={col} className={styles.fontMono}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(activeTab === "raw" ? preview.raw_data : preview.processed_data).map((row: Record<string, any>, rIdx: number) => (
                    <tr key={rIdx}>
                      <td><strong>{rIdx + 1}</strong></td>
                      {preview.columns.map((col: string) => {
                        const val = row[col];
                        const isNull = val === null || val === undefined;
                        const isNumeric = typeof val === "number";
                        
                        return (
                          <td 
                            key={col} 
                            className={isNumeric && activeTab === "processed" ? styles.processedValue : ""}
                          >
                            {isNull ? (
                              <span className={styles.nullValue}>? (Missing)</span>
                            ) : isNumeric ? (
                              val % 1 === 0 ? val : (val as number).toFixed(4)
                            ) : (
                              String(val)
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

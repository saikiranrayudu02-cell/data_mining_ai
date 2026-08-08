"use client";

import React, { useState } from "react";
import { useDataset } from "@/context/DatasetContext";
import { useToast } from "@/context/ToastContext";
import { apiService } from "@/services/api";
import { CompareResponse, AlgorithmType, EvaluationMode } from "@/types";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title as ChartTitle,
  Tooltip,
  Legend,
} from "chart.js";
import styles from "./page.module.css";
import commonStyles from "@/components/Common/Common.module.css";
import Link from "next/link";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ChartTitle,
  Tooltip,
  Legend
);

export default function ComparePage() {
  const { activeDatasetId, datasetMetadata } = useDataset();
  const { showToast } = useToast();
  
  const [evalMode, setEvalMode] = useState<EvaluationMode>("cross_validation");
  const [pctSplit, setPctSplit] = useState<number>(66);
  const [foldsCount, setFoldsCount] = useState<number>(10);
  const [randomSeed, setRandomSeed] = useState<number>(1);
  
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    if (!activeDatasetId || !datasetMetadata) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.compareClassifiers({
        dataset_id: activeDatasetId,
        target_attribute: datasetMetadata.class_attribute || "class",
        evaluation_mode: evalMode,
        percentage_split: pctSplit,
        folds: foldsCount,
        random_seed: randomSeed,
        hyperparameters: {
          KNN: { n_neighbors: 1, metric: "euclidean" },
          J48: { confidence_threshold: 0.25, min_instances: 2 },
          ID3: { min_instances: 1 }
        },
      });
      setComparison(res);
      showToast("WEKA multi-classifier benchmarking completed!", "success");
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Failed to run comparison.";
      setError(errMsg);
      showToast(errMsg, "error");
    } finally {
      setLoading(false);
    }
  };

  const handleExportCSV = async () => {
    if (!activeDatasetId || !datasetMetadata) return;
    setExporting(true);
    try {
      const blob = await apiService.exportComparisonCSV({
        dataset_id: activeDatasetId,
        target_attribute: datasetMetadata.class_attribute || "class",
        evaluation_mode: evalMode,
        percentage_split: pctSplit,
        folds: foldsCount,
        random_seed: randomSeed,
        hyperparameters: {
          KNN: { n_neighbors: 1, metric: "euclidean" },
          J48: { confidence_threshold: 0.25, min_instances: 2 },
          ID3: { min_instances: 1 }
        },
      });
      
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.setAttribute("download", `comparison_benchmark_${activeDatasetId.slice(0, 8)}.csv`);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      showToast("Comparison benchmark CSV exported successfully!", "success");
    } catch (err: unknown) {
      showToast(err instanceof Error ? err.message : "Failed to export CSV", "error");
    } finally {
      setExporting(false);
    }
  };

  if (!activeDatasetId || !datasetMetadata) {
    return (
      <div className={styles.container}>
        <div className={commonStyles.alert} style={{ backgroundColor: "rgba(239, 68, 68, 0.08)", border: "1px solid var(--color-error)" }}>
          <span>⚠️</span>
          <div>
            <h4 style={{ fontWeight: 600, marginBottom: "0.25rem" }}>Dataset not loaded</h4>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginBottom: "1rem" }}>
              Please load a dataset file first to run algorithm comparisons.
            </p>
            <Link href="/upload" className={`${commonStyles.btn} ${commonStyles.btnPrimary}`}>
              Upload Section &rarr;
            </Link>
          </div>
        </div>
      </div>
    );
  }

  const algKeys = comparison ? (Object.keys(comparison.results) as AlgorithmType[]) : [];
  const metricsList = comparison ? Object.values(comparison.results) : [];

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      y: {
        grid: { color: "rgba(255, 255, 255, 0.05)" },
        ticks: { color: "#94a3b8" },
      },
      x: {
        grid: { display: false },
        ticks: { color: "#94a3b8" },
      },
    },
  };

  const accuracyData = {
    labels: algKeys,
    datasets: [
      {
        data: metricsList.map((m) => m.accuracy * 100),
        backgroundColor: [
          "rgba(99, 102, 241, 0.5)",
          "rgba(168, 85, 247, 0.5)",
          "rgba(16, 185, 129, 0.5)",
          "rgba(245, 158, 11, 0.5)",
        ],
        borderColor: ["#6366f1", "#a855f7", "#10b981", "#f59e0b"],
        borderWidth: 1.5,
      },
    ],
  };

  const durationData = {
    labels: algKeys,
    datasets: [
      {
        data: metricsList.map((m) => m.execution_time_ms),
        backgroundColor: [
          "rgba(99, 102, 241, 0.5)",
          "rgba(168, 85, 247, 0.5)",
          "rgba(16, 185, 129, 0.5)",
          "rgba(245, 158, 11, 0.5)",
        ],
        borderColor: ["#6366f1", "#a855f7", "#10b981", "#f59e0b"],
        borderWidth: 1.5,
      },
    ],
  };

  const memoryData = {
    labels: algKeys,
    datasets: [
      {
        data: metricsList.map((m) => m.memory_used_mb || 0),
        backgroundColor: [
          "rgba(99, 102, 241, 0.5)",
          "rgba(168, 85, 247, 0.5)",
          "rgba(16, 185, 129, 0.5)",
          "rgba(245, 158, 11, 0.5)",
        ],
        borderColor: ["#6366f1", "#a855f7", "#10b981", "#f59e0b"],
        borderWidth: 1.5,
      },
    ],
  };

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.pageTitle}>Compare Classifiers</h1>
        <p className={styles.pageSubtitle}>
          Evaluate validation performance, execution speed, and memory footprint across J48, ID3, Naive Bayes, and k-NN on dataset <strong>{datasetMetadata.relation_name}</strong>
        </p>
      </div>

      <div className={commonStyles.card}>
        <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", alignItems: "center", marginBottom: "1rem" }}>
          <div style={{ flex: 1, minWidth: "220px" }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.35rem" }}>Evaluation Mode</label>
            <select
              value={evalMode}
              onChange={(e) => setEvalMode(e.target.value as EvaluationMode)}
              style={{ width: "100%", padding: "0.5rem", background: "var(--bg-tertiary)", border: "1px solid var(--border-color)", borderRadius: "4px", color: "var(--text-primary)" }}
            >
              <option value="cross_validation">10-fold Cross Validation</option>
              <option value="percentage_split">66% Percentage Split</option>
              <option value="training_set">Training Set</option>
            </select>
          </div>

          {evalMode === "cross_validation" && (
            <div style={{ width: "120px" }}>
              <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.35rem" }}>Folds</label>
              <input
                type="number"
                min={2}
                max={50}
                value={foldsCount}
                onChange={(e) => setFoldsCount(parseInt(e.target.value) || 10)}
                style={{ width: "100%", padding: "0.5rem", background: "var(--bg-tertiary)", border: "1px solid var(--border-color)", borderRadius: "4px", color: "var(--text-primary)" }}
              />
            </div>
          )}

          {evalMode === "percentage_split" && (
            <div style={{ width: "120px" }}>
              <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.35rem" }}>Train %</label>
              <input
                type="number"
                min={1}
                max={99}
                value={pctSplit}
                onChange={(e) => setPctSplit(parseFloat(e.target.value) || 66)}
                style={{ width: "100%", padding: "0.5rem", background: "var(--bg-tertiary)", border: "1px solid var(--border-color)", borderRadius: "4px", color: "var(--text-primary)" }}
              />
            </div>
          )}

          <div style={{ width: "120px" }}>
            <label style={{ fontSize: "0.85rem", color: "var(--text-secondary)", display: "block", marginBottom: "0.35rem" }}>Seed</label>
            <input
              type="number"
              value={randomSeed}
              onChange={(e) => setRandomSeed(parseInt(e.target.value) || 1)}
              style={{ width: "100%", padding: "0.5rem", background: "var(--bg-tertiary)", border: "1px solid var(--border-color)", borderRadius: "4px", color: "var(--text-primary)" }}
            />
          </div>
        </div>

        <div className={styles.compareControl}>
          <div className={styles.controlText}>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", margin: 0 }}>
              Run J48, ID3, Naive Bayes, and k-NN simultaneously using identical evaluation seeds and class attribute.
            </p>
          </div>
          <div style={{ display: "flex", gap: "1rem" }}>
            <button
              onClick={handleCompare}
              disabled={loading}
              className={`${commonStyles.btn} ${commonStyles.btnPrimary}`}
            >
              {loading ? "Computing Comparisons..." : "Run Benchmark Matrix"}
            </button>
            {comparison && (
              <button
                onClick={handleExportCSV}
                disabled={exporting}
                className={`${commonStyles.btn} ${commonStyles.btnSecondary}`}
              >
                {exporting ? "Exporting..." : "💾 Export CSV Report"}
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className={`${commonStyles.alert} ${commonStyles.alertError}`} style={{ marginTop: "1.5rem" }}>
            <span>⚠️</span>
            <div>{error}</div>
          </div>
        )}
      </div>

      {loading && (
        <div className={commonStyles.spinnerWrapper}>
          <div className={commonStyles.spinner}></div>
          <span className={commonStyles.spinnerText}>Running cross-validation evaluations...</span>
        </div>
      )}

      {comparison && !loading && (
        <div className={styles.resultsLayout}>
          {/* Best/Worst Performers widgets */}
          {comparison.best_algorithm && (
            <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
              <div className={commonStyles.card} style={{ flex: 1, minWidth: "250px", borderLeft: "4px solid var(--color-success)", padding: "1.25rem 1.5rem" }}>
                <h4 style={{ color: "var(--color-success)", fontSize: "0.75rem", textTransform: "uppercase", fontWeight: 700 }}>🏆 Best Performer</h4>
                <h2 style={{ margin: "0.25rem 0", color: "var(--text-primary)" }}>{comparison.best_algorithm}</h2>
                <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                  Accuracy: {(comparison.results[comparison.best_algorithm as AlgorithmType].accuracy * 100).toFixed(1)}% | Kappa: {comparison.results[comparison.best_algorithm as AlgorithmType].cohen_kappa.toFixed(4)}
                </span>
              </div>
              <div className={commonStyles.card} style={{ flex: 1, minWidth: "250px", borderLeft: "4px solid var(--color-error)", padding: "1.25rem 1.5rem" }}>
                <h4 style={{ color: "var(--color-error)", fontSize: "0.75rem", textTransform: "uppercase", fontWeight: 700 }}>📉 Lowest Performer</h4>
                <h2 style={{ margin: "0.25rem 0", color: "var(--text-primary)" }}>{comparison.worst_algorithm}</h2>
                <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                  Accuracy: {(comparison.results[comparison.worst_algorithm as AlgorithmType].accuracy * 100).toFixed(1)}% | Kappa: {comparison.results[comparison.worst_algorithm as AlgorithmType].cohen_kappa.toFixed(4)}
                </span>
              </div>
            </div>
          )}

          {/* Ranking Rationale Card */}
          {comparison.ranking_reason && (
            <div className={commonStyles.card} style={{ backgroundColor: "rgba(99, 102, 241, 0.05)", border: "1px solid var(--color-primary)", padding: "1rem 1.25rem", marginBottom: "1rem" }}>
              <h4 style={{ margin: "0 0 0.25rem 0", color: "var(--color-primary)", fontSize: "0.85rem" }}>Ranking Method Rationale</h4>
              <p style={{ margin: 0, color: "var(--text-secondary)", fontSize: "0.85rem" }}>{comparison.ranking_reason}</p>
            </div>
          )}

          {/* Charts grid */}
          <div className={styles.chartsGrid}>
            <div className={styles.chartCard}>
              <h3 className={styles.chartTitle}>Accuracy Benchmark (%)</h3>
              <div className={styles.chartContainer}>
                <Bar data={accuracyData} options={chartOptions} />
              </div>
            </div>

            <div className={styles.chartCard}>
              <h3 className={styles.chartTitle}>Training Run Times (ms)</h3>
              <div className={styles.chartContainer}>
                <Bar data={durationData} options={chartOptions} />
              </div>
            </div>

            <div className={styles.chartCard}>
              <h3 className={styles.chartTitle}>Memory Usage delta (MB)</h3>
              <div className={styles.chartContainer}>
                <Bar data={memoryData} options={chartOptions} />
              </div>
            </div>
          </div>

          {/* Table Details */}
          <div className={styles.tableCard}>
            <h3 className={styles.tableTitle}>Comparative Matrix Rankings</h3>
            <div className={styles.tableWrapper}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Algorithm</th>
                    <th>Accuracy</th>
                    <th>F1-Score</th>
                    <th>Kappa</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>Memory Delta</th>
                    <th>Run Time</th>
                  </tr>
                </thead>
                <tbody>
                  {algKeys.map((key) => {
                    const res = comparison.results[key];
                    return (
                      <tr key={key} style={{ backgroundColor: key === comparison.best_algorithm ? "rgba(16, 185, 129, 0.02)" : "transparent" }}>
                        <td>
                          <strong>{key}</strong>
                        </td>
                        <td>{(res.accuracy * 100).toFixed(1)}%</td>
                        <td>{(res.f1_score * 100).toFixed(1)}%</td>
                        <td>{res.cohen_kappa.toFixed(4)}</td>
                        <td>{res.mae !== undefined ? res.mae.toFixed(4) : "0.0000"}</td>
                        <td>{res.rmse !== undefined ? res.rmse.toFixed(4) : "0.0000"}</td>
                        <td>{res.memory_used_mb ? `${res.memory_used_mb.toFixed(3)} MB` : "0.0 MB"}</td>
                        <td>{res.execution_time_ms} ms</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

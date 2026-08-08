"use client";

import React, { useState, useEffect } from "react";
import { useDataset } from "@/context/DatasetContext";
import { useToast } from "@/context/ToastContext";
import { apiService } from "@/services/api";
import { ClassifyResponse, AlgorithmType, EvaluationMode } from "@/types";
import { WekaTreeVisualizer } from "@/components/WekaTree/WekaTreeVisualizer";
import styles from "./page.module.css";
import commonStyles from "@/components/Common/Common.module.css";
import Link from "next/link";

type ResultTab = "summary" | "detailed" | "confusion" | "roc" | "tree" | "rules" | "entropy" | "raw";

export default function ClassifyPage() {
  const { activeDatasetId, datasetMetadata } = useDataset();
  const { showToast } = useToast();
  
  const [algorithm, setAlgorithm] = useState<AlgorithmType>("J48");
  const [targetAttribute, setTargetAttribute] = useState<string>("");
  const [evalMode, setEvalMode] = useState<EvaluationMode>("cross_validation");
  const [pctSplit, setPctSplit] = useState<number>(66);
  const [foldsCount, setFoldsCount] = useState<number>(10);
  const [randomSeed, setRandomSeed] = useState<number>(1);
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ResultTab>("summary");

  // Algorithm-specific parameters
  const [kValue, setKValue] = useState<number>(1); // Weka default IBk is k=1
  const [distanceMetric, setDistanceMetric] = useState<string>("euclidean");
  const [confidence, setConfidence] = useState<number>(0.25);
  const [minInstances, setMinInstances] = useState<number>(2);
  const [pruning, setPruning] = useState<boolean>(true);
  const [maxDepth, setMaxDepth] = useState<string>("");

  useEffect(() => {
    if (datasetMetadata) {
      setTargetAttribute(datasetMetadata.class_attribute || (datasetMetadata.attributes.length > 0 ? datasetMetadata.attributes[datasetMetadata.attributes.length - 1].name : ""));
    }
  }, [datasetMetadata]);

  const handleTrain = async () => {
    if (!activeDatasetId || !datasetMetadata) return;
    setLoading(true);
    setError(null);
    try {
      const hyperparameters: Record<string, unknown> = {};
      if (algorithm === "KNN") {
        hyperparameters["n_neighbors"] = kValue;
        hyperparameters["metric"] = distanceMetric;
      } else if (algorithm === "J48") {
        hyperparameters["confidence_threshold"] = confidence;
        hyperparameters["min_instances"] = minInstances;
        hyperparameters["pruning"] = pruning;
        if (maxDepth) hyperparameters["max_depth"] = parseInt(maxDepth);
      } else if (algorithm === "ID3") {
        hyperparameters["min_instances"] = minInstances;
        if (maxDepth) hyperparameters["max_depth"] = parseInt(maxDepth);
      }

      const res = await apiService.trainClassifier({
        dataset_id: activeDatasetId,
        algorithm,
        target_attribute: targetAttribute,
        evaluation_mode: evalMode,
        percentage_split: pctSplit,
        folds: foldsCount,
        random_seed: randomSeed,
        hyperparameters
      });
      setResult(res);
      showToast(`Weka classification completed using ${algorithm}!`, "success");
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Failed to execute classification.";
      setError(errMsg);
      showToast(errMsg, "error");
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format: "pdf" | "html" | "json") => {
    if (!result || !activeDatasetId) return;
    try {
      showToast(`Generating ${format.toUpperCase()} report...`, "info");
      const res = await apiService.exportReport({
        dataset_id: activeDatasetId,
        model_id: result.model_id,
        format,
        include_metrics: true,
        include_rules: true,
        include_tree: true
      });
      
      const link = document.createElement("a");
      link.href = res.download_url;
      link.setAttribute("download", res.filename);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      showToast(`${format.toUpperCase()} report downloaded successfully!`, "success");
    } catch (err: unknown) {
      showToast(err instanceof Error ? err.message : "Failed to export report", "error");
    }
  };

  const copyWekaConsole = () => {
    if (result?.metrics.raw_weka_output) {
      navigator.clipboard.writeText(result.metrics.raw_weka_output);
      showToast("Raw Weka report text copied to clipboard!", "success");
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
              Please upload an ARFF dataset file first to configure and run classification.
            </p>
            <Link href="/upload" className={`${commonStyles.btn} ${commonStyles.btnPrimary}`}>
              Upload ARFF File &rarr;
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.pageHeader}>
        <h1 className={styles.pageTitle}>WEKA Classifier Workbench</h1>
        <p className={styles.pageSubtitle}>
          Configure test evaluation modes, parameters, and execute Weka-style classification on dataset: <strong>{datasetMetadata.relation_name}</strong>
        </p>
      </div>

      {/* WEKA Configuration Panel */}
      <div className={commonStyles.card} style={{ marginBottom: "2rem" }}>
        <h2 className={styles.sectionTitle}>1. Classifier & Evaluation Settings</h2>
        
        <div className={styles.settingsGrid}>
          
          {/* Classifier Selection */}
          <div className={`${styles.formGroup} ${styles.fullWidth}`}>
            <label className={styles.label}>Classifier Algorithm</label>
            <div className={styles.algoSelector}>
              {(["J48", "ID3", "NaiveBayes", "KNN"] as AlgorithmType[]).map((alg) => (
                <button
                  key={alg}
                  type="button"
                  onClick={() => setAlgorithm(alg)}
                  className={`${styles.algoBtn} ${algorithm === alg ? styles.algoBtnActive : ""}`}
                >
                  {alg === "J48" ? "J48 (C4.5)" : alg === "KNN" ? "k-NN (IBk)" : alg}
                </button>
              ))}
            </div>
          </div>

          {/* Class Attribute Selector */}
          <div className={styles.formGroup}>
            <label className={styles.label}>Class Attribute (Target)</label>
            <select
              value={targetAttribute}
              onChange={(e) => setTargetAttribute(e.target.value)}
              className={styles.selectInput}
            >
              {datasetMetadata.attributes.map((attr) => (
                <option key={attr.name} value={attr.name}>
                  {attr.name} ({attr.type})
                </option>
              ))}
            </select>
          </div>

          {/* Test Options */}
          <div className={styles.formGroup}>
            <label className={styles.label}>Test Evaluation Mode</label>
            <select
              value={evalMode}
              onChange={(e) => setEvalMode(e.target.value as EvaluationMode)}
              className={styles.selectInput}
            >
              <option value="cross_validation">Cross-validation (Default 10-fold)</option>
              <option value="percentage_split">Percentage split (Default 66% train)</option>
              <option value="training_set">Use training set</option>
            </select>
          </div>

          {/* Dynamic mode inputs */}
          {evalMode === "cross_validation" && (
            <div className={styles.formGroup}>
              <label className={styles.label}>Folds Count</label>
              <input
                type="number"
                min={2}
                max={50}
                value={foldsCount}
                onChange={(e) => setFoldsCount(parseInt(e.target.value) || 10)}
                className={styles.numberInput}
              />
            </div>
          )}

          {evalMode === "percentage_split" && (
            <div className={styles.formGroup}>
              <label className={styles.label}>Train Percentage (%)</label>
              <input
                type="number"
                min={1}
                max={99}
                value={pctSplit}
                onChange={(e) => setPctSplit(parseFloat(e.target.value) || 66)}
                className={styles.numberInput}
              />
            </div>
          )}

          <div className={styles.formGroup}>
            <label className={styles.label}>Random Seed</label>
            <input
              type="number"
              value={randomSeed}
              onChange={(e) => setRandomSeed(parseInt(e.target.value) || 1)}
              className={styles.numberInput}
            />
          </div>

        </div>

        {/* Algorithm Specific Hyperparameters */}
        <div className={styles.paramCard}>
          <h4 style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "0.75rem", fontSize: "0.9rem" }}>
            {algorithm} Configuration Parameters
          </h4>
          <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", alignItems: "center" }}>
            {algorithm === "J48" && (
              <>
                <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--text-secondary)", fontSize: "0.88rem", cursor: "pointer" }}>
                  <input type="checkbox" checked={pruning} onChange={(e) => setPruning(e.target.checked)} style={{ accentColor: "#6366f1", width: "16px", height: "16px" }} />
                  Subtree Pruning
                </label>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ fontSize: "0.88rem", color: "var(--text-secondary)" }}>Confidence Factor:</span>
                  <input
                    type="number"
                    step="0.05"
                    min="0.01"
                    max="0.5"
                    value={confidence}
                    onChange={(e) => setConfidence(parseFloat(e.target.value) || 0.25)}
                    className={styles.numberInput}
                    style={{ width: "90px" }}
                  />
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ fontSize: "0.88rem", color: "var(--text-secondary)" }}>Min Num Obj:</span>
                  <input
                    type="number"
                    min="1"
                    value={minInstances}
                    onChange={(e) => setMinInstances(parseInt(e.target.value) || 2)}
                    className={styles.numberInput}
                    style={{ width: "80px" }}
                  />
                </div>
              </>
            )}

            {algorithm === "ID3" && (
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ fontSize: "0.88rem", color: "var(--text-secondary)" }}>Min Instances per Leaf:</span>
                <input
                  type="number"
                  min="1"
                  value={minInstances}
                  onChange={(e) => setMinInstances(parseInt(e.target.value) || 1)}
                  className={styles.numberInput}
                  style={{ width: "80px" }}
                />
              </div>
            )}

            {algorithm === "KNN" && (
              <>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ fontSize: "0.88rem", color: "var(--text-secondary)" }}>K Value (KNN):</span>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={kValue}
                    onChange={(e) => setKValue(parseInt(e.target.value) || 1)}
                    className={styles.numberInput}
                    style={{ width: "80px" }}
                  />
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ fontSize: "0.88rem", color: "var(--text-secondary)" }}>Distance Metric:</span>
                  <select
                    value={distanceMetric}
                    onChange={(e) => setDistanceMetric(e.target.value)}
                    className={styles.selectInput}
                    style={{ width: "140px" }}
                  >
                    <option value="euclidean">Euclidean</option>
                    <option value="manhattan">Manhattan</option>
                    <option value="minkowski">Minkowski</option>
                  </select>
                </div>
              </>
            )}

            {algorithm === "NaiveBayes" && (
              <span style={{ color: "var(--text-secondary)", fontSize: "0.88rem" }}>
                Standard Naive Bayes algorithm (calculates prior & conditional probabilities based on Bayes theorem).
              </span>
            )}
          </div>
        </div>

        <div style={{ marginTop: "1.75rem", display: "flex", justifyContent: "flex-end" }}>
          <button
            onClick={handleTrain}
            disabled={loading}
            className={styles.startBtn}
          >
            {loading ? "Classifying..." : "⚡ Start Classification"}
          </button>
        </div>

        {error && (
          <div className={`${commonStyles.alert} ${commonStyles.alertError}`} style={{ marginTop: "1rem" }}>
            <span>⚠️</span>
            <div>{error}</div>
          </div>
        )}
      </div>

      {loading && (
        <div className={commonStyles.spinnerWrapper}>
          <div className={commonStyles.spinner}></div>
          <span className={commonStyles.spinnerText}>Executing WEKA classification pipeline...</span>
        </div>
      )}

      {/* WEKA Classification Results Section */}
      {result && !loading && (
        <div className={commonStyles.card}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem", marginBottom: "1.5rem" }}>
            <div>
              <h2 className={styles.sectionTitle}>Classification Results ({result.algorithm})</h2>
              <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", margin: 0 }}>
                Target: <strong>{targetAttribute}</strong> | Evaluation: <strong>{evalMode}</strong>
              </p>
            </div>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button onClick={() => handleExport("pdf")} className={`${commonStyles.btn} ${commonStyles.btnSecondary}`} style={{ fontSize: "0.8rem" }}>
                📄 PDF
              </button>
              <button onClick={() => handleExport("html")} className={`${commonStyles.btn} ${commonStyles.btnSecondary}`} style={{ fontSize: "0.8rem" }}>
                🌐 HTML
              </button>
              <button onClick={() => handleExport("json")} className={`${commonStyles.btn} ${commonStyles.btnSecondary}`} style={{ fontSize: "0.8rem" }}>
                📁 JSON
              </button>
            </div>
          </div>

          {/* 8 Tabs Navigation */}
          <div className={styles.tabsBar}>
            {[
              { id: "summary", label: "Summary" },
              { id: "detailed", label: "Detailed Accuracy" },
              { id: "confusion", label: "Confusion Matrix" },
              { id: "roc", label: "ROC Curve" },
              { id: "tree", label: "Decision Tree" },
              { id: "rules", label: "Decision Rules" },
              { id: "entropy", label: "Entropy & Gain" },
              { id: "raw", label: "Raw Weka Output" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as ResultTab)}
                className={`${styles.tabBtn} ${activeTab === tab.id ? styles.tabBtnActive : ""}`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab 1: Summary */}
          {activeTab === "summary" && (
            <div>
              <div className={styles.metricsGrid}>
                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Correctly Classified</div>
                  <div className={styles.metricValue} style={{ color: "var(--color-success)" }}>
                    {result.metrics.correctly_classified_instances || 0}
                  </div>
                  <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                    {(result.metrics.correctly_classified_pct || (result.metrics.accuracy * 100)).toFixed(2)}%
                  </span>
                </div>

                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Incorrectly Classified</div>
                  <div className={styles.metricValue} style={{ color: "var(--color-error)" }}>
                    {result.metrics.incorrectly_classified_instances || 0}
                  </div>
                  <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                    {(result.metrics.incorrectly_classified_pct || ((1 - result.metrics.accuracy) * 100)).toFixed(2)}%
                  </span>
                </div>

                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Kappa Statistic</div>
                  <div className={styles.metricValue}>
                    {result.metrics.cohen_kappa.toFixed(4)}
                  </div>
                </div>

                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Mean Absolute Error (MAE)</div>
                  <div className={styles.metricValue}>
                    {result.metrics.mae?.toFixed(4) || "0.0000"}
                  </div>
                </div>

                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Root Mean Squared Error</div>
                  <div className={styles.metricValue}>
                    {result.metrics.rmse?.toFixed(4) || "0.0000"}
                  </div>
                </div>

                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Relative Absolute Error</div>
                  <div className={styles.metricValue}>
                    {result.metrics.rae?.toFixed(2) || "0.00"}%
                  </div>
                </div>

                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Root Relative Sq. Error</div>
                  <div className={styles.metricValue}>
                    {result.metrics.rrse?.toFixed(2) || "0.00"}%
                  </div>
                </div>

                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Total Instances</div>
                  <div className={styles.metricValue}>
                    {result.metrics.total_instances || datasetMetadata.num_instances}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab 2: Detailed Accuracy */}
          {activeTab === "detailed" && (
            <div className={styles.tableWrapper}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>TP Rate</th>
                    <th>FP Rate</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F-Measure</th>
                    <th>MCC</th>
                    <th>ROC Area</th>
                    <th>PRC Area</th>
                  </tr>
                </thead>
                <tbody>
                  {result.metrics.detailed_accuracy_by_class?.map((cls, idx) => (
                    <tr key={idx}>
                      <td><strong>{cls.class_name}</strong></td>
                      <td>{cls.tp_rate.toFixed(3)}</td>
                      <td>{cls.fp_rate.toFixed(3)}</td>
                      <td>{cls.precision.toFixed(3)}</td>
                      <td>{cls.recall.toFixed(3)}</td>
                      <td>{cls.f_measure.toFixed(3)}</td>
                      <td>{cls.mcc.toFixed(3)}</td>
                      <td>{cls.roc_area.toFixed(3)}</td>
                      <td>{cls.prc_area.toFixed(3)}</td>
                    </tr>
                  ))}
                  {result.metrics.weighted_accuracy && (
                    <tr style={{ fontWeight: 700, backgroundColor: "var(--bg-tertiary)" }}>
                      <td>{result.metrics.weighted_accuracy.class_name}</td>
                      <td>{result.metrics.weighted_accuracy.tp_rate.toFixed(3)}</td>
                      <td>{result.metrics.weighted_accuracy.fp_rate.toFixed(3)}</td>
                      <td>{result.metrics.weighted_accuracy.precision.toFixed(3)}</td>
                      <td>{result.metrics.weighted_accuracy.recall.toFixed(3)}</td>
                      <td>{result.metrics.weighted_accuracy.f_measure.toFixed(3)}</td>
                      <td>{result.metrics.weighted_accuracy.mcc.toFixed(3)}</td>
                      <td>{result.metrics.weighted_accuracy.roc_area.toFixed(3)}</td>
                      <td>{result.metrics.weighted_accuracy.prc_area.toFixed(3)}</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}

          {/* Tab 3: Confusion Matrix */}
          {activeTab === "confusion" && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.5rem" }}>
              <div>
                <h3 className={styles.subTitle}>WEKA Confusion Matrix</h3>
                <div style={{ background: "var(--bg-tertiary)", padding: "1.25rem", borderRadius: "8px", fontFamily: "monospace" }}>
                  <table style={{ borderCollapse: "collapse", width: "100%", textAlign: "center" }}>
                    <thead>
                      <tr>
                        <th style={{ borderBottom: "1px solid var(--border-color)", padding: "0.5rem" }}>Actual \ Pred</th>
                        {result.metrics.confusion_matrix.labels.map((l) => (
                          <th key={l} style={{ borderBottom: "1px solid var(--border-color)", padding: "0.5rem" }}>{l}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.metrics.confusion_matrix.matrix.map((row, rIdx) => (
                        <tr key={rIdx}>
                          <td style={{ fontWeight: 700, textAlign: "left", padding: "0.5rem", borderRight: "1px solid var(--border-color)" }}>
                            {result.metrics.confusion_matrix.labels[rIdx]}
                          </td>
                          {row.map((val, cIdx) => (
                            <td
                              key={cIdx}
                              style={{
                                padding: "0.5rem",
                                backgroundColor: rIdx === cIdx ? "rgba(16, 185, 129, 0.15)" : "transparent",
                                color: rIdx === cIdx ? "var(--color-success)" : "var(--text-primary)",
                                fontWeight: rIdx === cIdx ? 700 : 400
                              }}
                            >
                              {val}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {result.metrics.confusion_matrix_plot_url && (
                <div>
                  <h3 className={styles.subTitle}>Heatmap Visualization</h3>
                  <div className={styles.plotImgWrapper}>
                    <img src={result.metrics.confusion_matrix_plot_url} alt="Confusion Matrix" className={styles.plotImg} />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab 4: ROC Curve */}
          {activeTab === "roc" && (
            <div>
              <h3 className={styles.subTitle}>Multiclass One-vs-Rest ROC Curve</h3>
              {result.metrics.roc_curve_plot_url ? (
                <div className={styles.plotImgWrapper} style={{ maxWidth: "600px", margin: "0 auto" }}>
                  <img src={result.metrics.roc_curve_plot_url} alt="ROC Curve" className={styles.plotImg} />
                </div>
              ) : (
                <p style={{ color: "var(--text-secondary)" }}>ROC curve is generated for probabilistic classifiers.</p>
              )}
            </div>
          )}

          {/* Tab 5: Decision Tree Visualizer */}
          {activeTab === "tree" && (
            <div>
              <h3 className={styles.subTitle}>WEKA Explorer Decision Tree Visualizer</h3>
              <WekaTreeVisualizer
                tree={result.tree ?? null}
                wekaTreeText={result.metrics.weka_tree_text ?? null}
                treePngUrl={result.tree_png_url ?? null}
                algorithm={result.algorithm}
                depth={result.tree_depth ?? null}
                leafNodes={result.tree_leaf_nodes ?? null}
                splitNodes={result.tree_internal_nodes ?? null}
              />
            </div>
          )}

          {/* Tab 6: Decision Rules */}
          {activeTab === "rules" && (
            <div>
              <h3 className={styles.subTitle}>Extracted IF-THEN Rules ({result.rules.length} pathways)</h3>
              {result.rules.length > 0 ? (
                <div className={styles.rulesList}>
                  {result.rules.map((rule, idx) => (
                    <div key={idx} className={styles.ruleCard}>
                      <span className={styles.ruleBadge}>Rule #{idx + 1}</span>
                      <pre className={styles.ruleText}>{rule}</pre>
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ color: "var(--text-secondary)" }}>Rules are extracted for decision tree algorithms (J48, ID3).</p>
              )}
            </div>
          )}

          {/* Tab 7: Entropy & Info Gain */}
          {activeTab === "entropy" && (
            <div>
              <h3 className={styles.subTitle}>Candidate Split Attributes Evaluation (Entropy & Gain Ratio)</h3>
              {result.metrics.entropy_stats && result.metrics.entropy_stats.length > 0 ? (
                <div className={styles.tableWrapper}>
                  <table className={styles.table}>
                    <thead>
                      <tr>
                        <th>Attribute Name</th>
                        <th>Base Entropy</th>
                        <th>Information Gain</th>
                        <th>Split Information</th>
                        <th>Gain Ratio</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.metrics.entropy_stats.map((s, idx) => (
                        <tr key={idx}>
                          <td><strong>{s.attribute_name}</strong></td>
                          <td>{s.entropy.toFixed(4)}</td>
                          <td>{s.info_gain.toFixed(4)}</td>
                          <td>{s.split_info !== null && s.split_info !== undefined ? s.split_info.toFixed(4) : "N/A"}</td>
                          <td>{s.gain_ratio !== null && s.gain_ratio !== undefined ? s.gain_ratio.toFixed(4) : "N/A"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{ color: "var(--text-secondary)" }}>Entropy and Information Gain tables are calculated for J48 and ID3.</p>
              )}
            </div>
          )}

          {/* Tab 8: Raw Weka Output */}
          {activeTab === "raw" && (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                <h3 className={styles.subTitle} style={{ margin: 0 }}>Raw WEKA Explorer Text Report</h3>
                <button onClick={copyWekaConsole} className={`${commonStyles.btn} ${commonStyles.btnSecondary}`} style={{ fontSize: "0.8rem" }}>
                  📋 Copy Output
                </button>
              </div>
              <pre style={{ background: "#090d16", color: "#4ade80", padding: "1.5rem", borderRadius: "8px", overflowX: "auto", fontFamily: "Courier New, monospace", fontSize: "0.85rem", lineHeight: "1.45" }}>
                {result.metrics.raw_weka_output || "Raw WEKA output text unavailable."}
              </pre>
            </div>
          )}

        </div>
      )}
    </div>
  );
}

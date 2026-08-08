"use client";

import React, { useState } from "react";
import { apiService } from "@/services/api";
import { ClassifyResponse, AlgorithmType } from "@/types";

export default function RulesPage() {
  const datasetId = "mock-id-123";
  const [algorithm, setAlgorithm] = useState<AlgorithmType>("J48");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFetchRules = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.trainClassifier({
        dataset_id: datasetId,
        algorithm,
        target_attribute: "play",
        test_split: 0.2
      });
      setResult(res);
      setExportUrl(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load rules");
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format: "pdf" | "json") => {
    if (!result) return;
    setExporting(true);
    try {
      const res = await apiService.exportReport({
        dataset_id: datasetId,
        model_id: result.model_id,
        format,
        include_metrics: true,
        include_rules: true,
        include_tree: true
      });
      setExportUrl(res.download_url);
    } catch (err: unknown) {
      setError(err instanceof Error ? "Export failed: " + err.message : "Export failed");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="rules-page">
      <div className="page-header">
        <h1 className="page-title">Rule Extractor & Decision Trees</h1>
        <p className="page-subtitle">Extract human-readable IF-THEN rules and explore hierarchical decision paths.</p>
      </div>

      <div className="control-card card">
        <div className="control-row">
          <div className="control-group">
            <label className="control-label">Decision Tree Algorithm</label>
            <select 
              value={algorithm} 
              onChange={(e) => setAlgorithm(e.target.value as AlgorithmType)}
              className="control-select"
            >
              <option value="J48">J48 (C4.5) Tree</option>
              <option value="ID3">ID3 Tree</option>
            </select>
          </div>
          <button 
            onClick={handleFetchRules} 
            disabled={loading} 
            className="btn btn-primary"
          >
            {loading ? "Extracting..." : "Generate Tree & Rules"}
          </button>
        </div>

        {error && <div className="error-alert">{error}</div>}
      </div>

      {result && (
        <div className="visualization-layout">
          {/* Rules List Panel */}
          <div className="rules-panel card">
            <div className="panel-header">
              <h2 className="panel-title">Extracted IF-THEN Rules</h2>
              <div className="export-actions">
                <button 
                  onClick={() => handleExport("pdf")} 
                  disabled={exporting}
                  className="btn btn-secondary btn-sm"
                >
                  {exporting ? "Generating PDF..." : "Export PDF"}
                </button>
                <button 
                  onClick={() => handleExport("json")} 
                  disabled={exporting}
                  className="btn btn-secondary btn-sm"
                >
                  Export JSON
                </button>
              </div>
            </div>

            {exportUrl && (
              <div className="export-success">
                Report generated! <a href={exportUrl} className="download-link" target="_blank" rel="noreferrer">Click here to download</a>
              </div>
            )}

            <div className="rules-list">
              {result.rules.map((rule, idx) => (
                <div className="rule-item" key={idx}>
                  <span className="rule-badge">Rule {idx + 1}</span>
                  <code className="rule-code">{rule}</code>
                </div>
              ))}
            </div>
          </div>

          {/* Decision Tree Node Layout Panel */}
          <div className="tree-panel card">
            <h2 className="panel-title">Decision Tree Node Visualizer</h2>
            <div className="canvas-placeholder">
              <div className="tree-root-mock">
                {result.tree?.nodes.map((node) => (
                  <div className={`mock-node ${node.type}`} key={node.id}>
                    <span className="node-id">ID: {node.id}</span>
                    <span className="node-label">{node.label}</span>
                    <span className="node-type">{(node.data?.type as string) || "split"}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .rules-page {
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .page-title {
          font-size: 2.2rem;
          font-weight: 700;
          margin-bottom: 0.5rem;
        }

        .page-subtitle {
          color: var(--text-secondary);
        }

        .card {
          background-color: var(--bg-secondary);
          border: 1px solid var(--border-color);
          border-radius: var(--radius-md);
          padding: 2rem;
        }

        .control-card {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .control-row {
          display: flex;
          justify-content: space-between;
          align-items: flex-end;
          flex-wrap: wrap;
          gap: 1.5rem;
        }

        .control-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .control-label {
          font-size: 0.9rem;
          color: var(--text-secondary);
        }

        .control-select {
          background-color: var(--bg-primary);
          border: 1px solid var(--border-color);
          border-radius: var(--radius-sm);
          color: var(--text-primary);
          padding: 0.75rem;
          font-size: 0.95rem;
          min-width: 240px;
          outline: none;
        }

        .btn {
          padding: 0.75rem 1.5rem;
          border-radius: var(--radius-sm);
          font-size: 0.95rem;
          font-weight: 600;
          transition: var(--transition-fast);
        }

        .btn-primary {
          background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
          color: var(--text-primary);
        }

        .btn-primary:hover {
          opacity: 0.9;
        }

        .btn-secondary {
          background-color: var(--bg-tertiary);
          border: 1px solid var(--border-color);
          color: var(--text-primary);
        }

        .btn-secondary:hover {
          background-color: var(--border-color);
        }

        .btn-sm {
          padding: 0.4rem 0.8rem;
          font-size: 0.8rem;
        }

        .error-alert {
          padding: 0.75rem 1rem;
          background-color: rgba(239, 68, 68, 0.1);
          border: 1px solid var(--color-error);
          color: #f87171;
          border-radius: var(--radius-sm);
          font-size: 0.85rem;
        }

        .visualization-layout {
          display: grid;
          grid-template-columns: 1fr;
          gap: 2rem;
        }

        @media (min-width: 992px) {
          .visualization-layout {
            grid-template-columns: 1fr 1fr;
          }
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
          flex-wrap: wrap;
          gap: 1rem;
        }

        .panel-title {
          font-size: 1.3rem;
          font-weight: 600;
        }

        .export-actions {
          display: flex;
          gap: 0.5rem;
        }

        .export-success {
          background-color: rgba(16, 185, 129, 0.1);
          border: 1px solid var(--color-success);
          color: var(--color-success);
          padding: 0.75rem 1rem;
          border-radius: var(--radius-sm);
          font-size: 0.9rem;
          margin-bottom: 1.5rem;
        }

        .download-link {
          text-decoration: underline;
          font-weight: 600;
        }

        .rules-list {
          display: flex;
          flex-direction: column;
          gap: 1rem;
          max-height: 400px;
          overflow-y: auto;
        }

        .rule-item {
          background-color: var(--bg-primary);
          border: 1px solid var(--border-color);
          border-radius: var(--radius-sm);
          padding: 1rem;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .rule-badge {
          font-size: 0.75rem;
          font-weight: 600;
          color: var(--accent-secondary);
          text-transform: uppercase;
        }

        .rule-code {
          font-family: monospace;
          color: var(--text-primary);
          font-size: 0.9rem;
          white-space: pre-wrap;
        }

        .canvas-placeholder {
          background-color: var(--bg-primary);
          border: 1px solid var(--border-color);
          border-radius: var(--radius-sm);
          height: 400px;
          overflow: auto;
          position: relative;
          padding: 2rem;
        }

        .tree-root-mock {
          display: flex;
          flex-wrap: wrap;
          gap: 1rem;
          justify-content: center;
          align-items: center;
        }

        .mock-node {
          background-color: var(--bg-tertiary);
          border: 1px solid var(--border-color);
          border-radius: var(--radius-sm);
          padding: 0.75rem 1rem;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.25rem;
          min-width: 100px;
        }

        .mock-node.input {
          border-color: var(--accent-primary);
          box-shadow: 0 0 10px rgba(99, 102, 241, 0.2);
        }

        .mock-node.output {
          border-color: var(--color-success);
        }

        .node-id {
          font-size: 0.7rem;
          color: var(--text-muted);
        }

        .node-label {
          font-size: 0.9rem;
          font-weight: 600;
        }

        .node-type {
          font-size: 0.75rem;
          color: var(--text-secondary);
          text-transform: uppercase;
        }
      `}</style>
    </div>
  );
}

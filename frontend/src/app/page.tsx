"use client";

import React from "react";
import Link from "next/link";
import { useDataset } from "@/context/DatasetContext";
import styles from "./page.module.css";
import commonStyles from "@/components/Common/Common.module.css";

export default function Home() {
  const { activeDatasetId, datasetMetadata } = useDataset();

  return (
    <div className={styles.container}>
      <section className={styles.hero}>
        <h1 className={styles.title}>DataMine AI Classifier</h1>
        <p className={styles.subtitle}>
          A premium machine learning platform for Attribute-Relation File Format (.arff) classification, 
          decision tree modeling, and performance evaluation.
        </p>
      </section>

      {activeDatasetId && datasetMetadata ? (
        <section className={commonStyles.glassCard}>
          <h2 className={styles.stepTitle}>Active Dataset Loaded</h2>
          <p className={styles.stepDesc} style={{ marginBottom: "1.5rem" }}>
            The relation <strong>{datasetMetadata.relation_name}</strong> is currently active in memory. 
            You can proceed to preview rows, configure training models, or evaluate algorithm statistics.
          </p>
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            <Link href="/preview" className={`${commonStyles.btn} ${commonStyles.btnPrimary}`}>
              👁️ View Data Preview
            </Link>
            <Link href="/classify" className={`${commonStyles.btn} ${commonStyles.btnSecondary}`}>
              🧠 Train Classification Models
            </Link>
            <Link href="/compare" className={`${commonStyles.btn} ${commonStyles.btnSecondary}`}>
              ⚔️ Run Comparison Matrix
            </Link>
          </div>
        </section>
      ) : (
        <section className={commonStyles.alert} style={{ backgroundColor: "rgba(99, 102, 241, 0.08)", border: "1px solid rgba(99, 102, 241, 0.3)" }}>
          <span style={{ fontSize: "1.5rem" }}>💡</span>
          <div>
            <h4 style={{ fontWeight: 600, marginBottom: "0.25rem" }}>No active dataset detected</h4>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem" }}>
              To unlock full features, navigate to the upload section and choose an ARFF dataset file.
            </p>
          </div>
        </section>
      )}

      <section className={styles.workflowSection}>
        <h2 className={styles.sectionTitle}>Workflow Stages</h2>
        <div className={styles.grid}>
          <div className={`${commonStyles.card} ${commonStyles.cardHover}`}>
            <span className={styles.stepIcon}>📤</span>
            <h3 className={styles.stepTitle}>1. Upload ARFF Dataset</h3>
            <p className={styles.stepDesc}>
              Import and parse standard datasets. Instantly extract relation headers, names, and attribute specifications.
            </p>
            <Link href="/upload" className={styles.stepBtn}>
              Upload File &rarr;
            </Link>
          </div>

          <div className={`${commonStyles.card} ${commonStyles.cardHover}`}>
            <span className={styles.stepIcon}>👁️</span>
            <h3 className={styles.stepTitle}>2. Inspect & Preprocess</h3>
            <p className={styles.stepDesc}>
              Compare raw features with automatically preprocessed values (imputed means/modes, encoded labels, and normalized scalings).
            </p>
            <Link href="/preview" className={styles.stepBtn}>
              Preview Tables &rarr;
            </Link>
          </div>

          <div className={`${commonStyles.card} ${commonStyles.cardHover}`}>
            <span className={styles.stepIcon}>🧠</span>
            <h3 className={styles.stepTitle}>3. Train & Classify</h3>
            <p className={styles.stepDesc}>
              Train ID3, J48, Naive Bayes, or KNN models. Explore metrics, textual classification reports, and interactive React Flow tree graphs.
            </p>
            <Link href="/classify" className={styles.stepBtn}>
              Train Models &rarr;
            </Link>
          </div>

          <div className={`${commonStyles.card} ${commonStyles.cardHover}`}>
            <span className={styles.stepIcon}>⚔️</span>
            <h3 className={styles.stepTitle}>4. Compare Performance</h3>
            <p className={styles.stepDesc}>
              Run side-by-side checks on all algorithms. Generate Chart.js diagrams evaluating validation accuracy and run times.
            </p>
            <Link href="/compare" className={styles.stepBtn}>
              Compare Models &rarr;
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

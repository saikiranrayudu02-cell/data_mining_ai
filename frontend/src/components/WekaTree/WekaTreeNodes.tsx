import React from "react";
import { Handle, Position, NodeProps } from "@xyflow/react";
import styles from "./WekaTreeVisualizer.module.css";

export function SplitNodeComponent({ data }: NodeProps) {
  const featureLabel = typeof data.label === "string" ? data.label : String(data.feature || "Split Attribute");
  const isActive = Boolean(data.isActive);

  return (
    <div className={`${styles.splitNode} ${isActive ? styles.splitNodeActive : ""}`}>
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: "#38bdf8", width: 9, height: 9, border: "2px solid #0f172a" }}
      />
      <span className={styles.splitNodeBadge}>Split</span>
      <span>{featureLabel}</span>
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: "#38bdf8", width: 9, height: 9, border: "2px solid #0f172a" }}
      />
    </div>
  );
}

export function LeafNodeComponent({ data }: NodeProps) {
  const predictionLabel = typeof data.prediction === "string" ? data.prediction : String(data.label || "Leaf");
  const total = Number(data.total_instances || 0);
  const error = Number(data.error_instances || 0);
  const isActive = Boolean(data.isActive);

  return (
    <div className={`${styles.leafNode} ${isActive ? styles.leafNodeActive : ""}`}>
      <Handle
        type="target"
        position={Position.Top}
        style={{ background: "#10b981", width: 9, height: 9, border: "2px solid #0f172a" }}
      />
      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
        <span>🌿</span>
        <span>{predictionLabel}</span>
      </div>

      {total > 0 && (
        <span className={styles.leafCountBadge}>
          ({total.toFixed(0)}
          {error > 0 && <span className={styles.leafError}>/{error.toFixed(0)}</span>})
        </span>
      )}
    </div>
  );
}

export const wekaNodeTypes = {
  split: SplitNodeComponent,
  leaf: LeafNodeComponent,
  default: SplitNodeComponent,
  input: SplitNodeComponent,
  output: LeafNodeComponent
};

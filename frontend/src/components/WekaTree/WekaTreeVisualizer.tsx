"use client";

import React, { useState, useCallback, useMemo } from "react";
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  MarkerType,
  Node,
  Edge
} from "@xyflow/react";
import { wekaNodeTypes } from "./WekaTreeNodes";
import { DecisionTreeStructure } from "@/types";
import styles from "./WekaTreeVisualizer.module.css";

interface WekaTreeVisualizerProps {
  tree: DecisionTreeStructure | null;
  wekaTreeText?: string | null;
  treePngUrl?: string | null;
  algorithm?: string;
  depth?: number | null;
  leafNodes?: number | null;
  splitNodes?: number | null;
}

type ViewMode = "canvas" | "graphviz" | "ascii";

export function WekaTreeVisualizer({
  tree,
  wekaTreeText,
  treePngUrl,
  algorithm = "J48",
  depth,
  leafNodes,
  splitNodes
}: WekaTreeVisualizerProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("canvas");
  const [showMiniMap, setShowMiniMap] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);

  // Compute active highlight paths
  const { highlightedNodeIds, highlightedEdgeIds } = useMemo(() => {
    if (!tree || !activeNodeId) {
      return { highlightedNodeIds: new Set<string>(), highlightedEdgeIds: new Set<string>() };
    }

    const nodeIds = new Set<string>([activeNodeId]);
    const edgeIds = new Set<string>();

    // Trace ancestors
    let currentId = activeNodeId;
    while (currentId) {
      const parentEdge = tree.edges.find((e) => e.target === currentId);
      if (parentEdge) {
        edgeIds.add(parentEdge.id);
        nodeIds.add(parentEdge.source);
        currentId = parentEdge.source;
      } else {
        break;
      }
    }

    return { highlightedNodeIds: nodeIds, highlightedEdgeIds: edgeIds };
  }, [tree, activeNodeId]);

  // Format ReactFlow Nodes
  const initialNodes: Node[] = useMemo(() => {
    if (!tree?.nodes) return [];
    return tree.nodes.map((n) => {
      const isLeaf = n.type === "leaf" || n.data?.type === "leaf";
      const isActive = highlightedNodeIds.has(n.id);
      return {
        id: n.id,
        type: isLeaf ? "leaf" : "split",
        data: {
          ...n.data,
          label: n.label,
          isActive
        },
        position: n.position || { x: 0, y: 0 }
      };
    });
  }, [tree, highlightedNodeIds]);

  // Format ReactFlow Edges
  const initialEdges: Edge[] = useMemo(() => {
    if (!tree?.edges) return [];
    return tree.edges.map((e) => {
      const isHighlighted = highlightedEdgeIds.has(e.id);
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        type: "smoothstep",
        label: e.label,
        style: {
          stroke: isHighlighted ? "#a855f7" : "#64748b",
          strokeWidth: isHighlighted ? 3.5 : 2,
          transition: "all 0.2s ease"
        },
        labelStyle: {
          fill: isHighlighted ? "#f8fafc" : "#cbd5e1",
          fontWeight: 700,
          fontSize: 11
        },
        labelBgStyle: {
          fill: isHighlighted ? "#581c87" : "#0f172a",
          fillOpacity: 0.95,
          rx: 6,
          ry: 6
        },
        labelBgPadding: [8, 4] as [number, number],
        labelBgBorderRadius: 6,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: isHighlighted ? "#a855f7" : "#64748b",
          width: 16,
          height: 16
        }
      };
    });
  }, [tree, highlightedEdgeIds]);

  const handleNodeMouseEnter = useCallback((_: React.MouseEvent, node: Node) => {
    setActiveNodeId(node.id);
  }, []);

  const handleNodeMouseLeave = useCallback(() => {
    setActiveNodeId(null);
  }, []);

  const totalNodesCount = (leafNodes || 0) + (splitNodes || 0);

  if (!tree && !wekaTreeText && !treePngUrl) {
    return (
      <div className={styles.treeContainer} style={{ padding: "2rem", textAlign: "center", color: "var(--text-secondary)" }}>
        <p>No decision tree structure available for this classification model.</p>
      </div>
    );
  }

  return (
    <div className={`${styles.treeContainer} ${isFullscreen ? styles.treeContainerFullscreen : ""}`}>
      {/* WEKA Explorer Style Toolbar */}
      <div className={styles.treeHeader}>
        <div className={styles.statsGroup}>
          <span className={styles.statBadge}>
            🌴 Algorithm: <span className={styles.statValue}>{algorithm}</span>
          </span>
          {depth !== null && depth !== undefined && (
            <span className={styles.statBadge}>
              📏 Depth: <span className={styles.statValue}>{depth}</span>
            </span>
          )}
          {totalNodesCount > 0 && (
            <span className={styles.statBadge}>
              🌿 Nodes: <span className={styles.statValue}>{totalNodesCount}</span> ({splitNodes || 0} split, {leafNodes || 0} leaf)
            </span>
          )}
        </div>

        <div className={styles.viewModeGroup}>
          <button
            type="button"
            className={`${styles.viewBtn} ${viewMode === "canvas" ? styles.viewBtnActive : ""}`}
            onClick={() => setViewMode("canvas")}
          >
            🗺️ WEKA Interactive Tree
          </button>
          {treePngUrl && (
            <button
              type="button"
              className={`${styles.viewBtn} ${viewMode === "graphviz" ? styles.viewBtnActive : ""}`}
              onClick={() => setViewMode("graphviz")}
            >
              🖼️ Graphviz Vector Image
            </button>
          )}
          {wekaTreeText && (
            <button
              type="button"
              className={`${styles.viewBtn} ${viewMode === "ascii" ? styles.viewBtnActive : ""}`}
              onClick={() => setViewMode("ascii")}
            >
              📜 WEKA Text Output
            </button>
          )}
        </div>

        <div className={styles.controlsGroup}>
          {viewMode === "canvas" && (
            <button
              type="button"
              className={`${styles.actionBtn} ${showMiniMap ? styles.actionBtnActive : ""}`}
              onClick={() => setShowMiniMap(!showMiniMap)}
              title="Toggle Minimap View"
            >
              🗺️ {showMiniMap ? "Hide Map" : "Show Map"}
            </button>
          )}
          <button
            type="button"
            className={styles.actionBtn}
            onClick={() => setIsFullscreen(!isFullscreen)}
            title="Toggle Fullscreen"
          >
            {isFullscreen ? "🗗 Exit Fullscreen" : "🗖 Fullscreen"}
          </button>
        </div>
      </div>

      {/* Main View Area */}
      {viewMode === "canvas" && tree && (
        <div className={`${styles.canvasWrapper} ${isFullscreen ? styles.canvasWrapperFullscreen : ""}`}>
          <ReactFlow
            nodes={initialNodes}
            edges={initialEdges}
            nodeTypes={wekaNodeTypes}
            onNodeMouseEnter={handleNodeMouseEnter}
            onNodeMouseLeave={handleNodeMouseLeave}
            fitView
            minZoom={0.2}
            maxZoom={2.5}
            fitViewOptions={{ padding: 0.2 }}
          >
            <Background color="#1e293b" gap={20} size={1} />
            <Controls />
            {showMiniMap && (
              <MiniMap
                zoomable
                pannable
                nodeColor={(node) => (node.type === "leaf" ? "#10b981" : "#38bdf8")}
                maskColor="rgba(15, 23, 42, 0.75)"
                style={{
                  background: "#0f172a",
                  border: "1px solid rgba(255, 255, 255, 0.15)",
                  borderRadius: "8px"
                }}
              />
            )}
          </ReactFlow>
        </div>
      )}

      {viewMode === "graphviz" && treePngUrl && (
        <div className={styles.imageViewContainer}>
          <img src={treePngUrl} alt="WEKA Graphviz Decision Tree" className={styles.treeImage} />
          <div style={{ marginTop: "1rem" }}>
            <a
              href={treePngUrl}
              target="_blank"
              rel="noreferrer"
              className={styles.actionBtn}
              style={{ display: "inline-flex" }}
            >
              🔗 Open Full High-Res Image in New Tab
            </a>
          </div>
        </div>
      )}

      {viewMode === "ascii" && wekaTreeText && (
        <div className={styles.textViewContainer}>
          <div className={styles.wekaTerminal}>
            <pre>{wekaTreeText}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

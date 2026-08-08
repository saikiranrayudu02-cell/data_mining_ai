"use client";

import React from "react";
import { useDataset } from "@/context/DatasetContext";
import { useSidebar } from "@/context/SidebarContext";
import styles from "./Navbar.module.css";
import Link from "next/link";

export default function Navbar() {
  const { datasetMetadata } = useDataset();
  const { toggleSidebar } = useSidebar();

  return (
    <nav className={styles.navbar}>
      <div className={styles.leftSection}>
        <button 
          className={styles.hamburgerBtn} 
          onClick={toggleSidebar}
          aria-label="Toggle navigation menu"
        >
          ☰
        </button>
        <Link href="/" className={styles.logoSection}>
          <span className={styles.logoIcon}>📊</span>
          <span className={styles.logoText}>
            DataMine <span className={styles.logoHighlight}>AI</span>
          </span>
        </Link>
      </div>
      <div className={styles.statusSection}>
        {datasetMetadata && (
          <div className={styles.activeDatasetBadge} title={datasetMetadata.relation_name}>
            Active: {datasetMetadata.relation_name}
          </div>
        )}
        <div className={styles.statusIndicator}>
          <span className={styles.dot}></span>
          API Connected
        </div>
      </div>
    </nav>
  );
}

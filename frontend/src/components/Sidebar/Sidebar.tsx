"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useDataset } from "@/context/DatasetContext";
import { useSidebar } from "@/context/SidebarContext";
import styles from "./Sidebar.module.css";

export default function Sidebar() {
  const pathname = usePathname();
  const { activeDatasetId } = useDataset();
  const { isSidebarOpen, setSidebarOpen } = useSidebar();

  const navItems = [
    { name: "Dashboard", path: "/", icon: "🏠", requireDataset: false },
    { name: "Upload Dataset", path: "/upload", icon: "📤", requireDataset: false },
    { name: "Dataset Preview", path: "/preview", icon: "👁️", requireDataset: true },
    { name: "Train Classifier", path: "/classify", icon: "🧠", requireDataset: true },
    { name: "Compare Models", path: "/compare", icon: "⚔️", requireDataset: true },
  ];

  const handleNavClick = () => {
    if (window.innerWidth <= 1024) {
      setSidebarOpen(false);
    }
  };

  return (
    <>
      {/* Mobile overlay */}
      {isSidebarOpen && (
        <div 
          className={styles.sidebarOverlay} 
          onClick={() => setSidebarOpen(false)} 
          aria-label="Close sidebar"
        />
      )}
      <aside className={`${styles.sidebar} ${isSidebarOpen ? styles.open : ""}`}>
        <nav className={styles.navSection}>
        {navItems.map((item) => {
          const isActive = pathname === item.path;
          const isDisabled = item.requireDataset && !activeDatasetId;
          
          if (isDisabled) {
            return (
              <span 
                key={item.name} 
                className={`${styles.navLink}`}
                style={{ opacity: 0.3, cursor: "not-allowed" }}
                title="Please upload a dataset first"
              >
                <span className={styles.navIcon}>{item.icon}</span>
                {item.name}
              </span>
            );
          }

          return (
            <Link
              key={item.name}
              href={item.path}
              onClick={handleNavClick}
              className={`${styles.navLink} ${isActive ? styles.activeLink : ""}`}
            >
              <span className={styles.navIcon}>{item.icon}</span>
              {item.name}
            </Link>
          );
        })}
      </nav>
      <div className={styles.divider}></div>
      <div className={styles.sidebarFooter}>
        <p>DataMine AI v1.0.0</p>
        <p>FastAPI & Next.js Engine</p>
      </div>
    </aside>
    </>
  );
}

"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";
import { DatasetMetadata } from "@/types";

interface DatasetContextType {
  activeDatasetId: string | null;
  datasetMetadata: DatasetMetadata | null;
  setActiveDataset: (id: string, metadata: DatasetMetadata) => void;
  clearDataset: () => void;
}

const DatasetContext = createContext<DatasetContextType | undefined>(undefined);

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [activeDatasetId, setActiveDatasetId] = useState<string | null>(null);
  const [datasetMetadata, setDatasetMetadata] = useState<DatasetMetadata | null>(null);

  const setActiveDataset = (id: string, metadata: DatasetMetadata) => {
    setActiveDatasetId(id);
    setDatasetMetadata(metadata);
  };

  const clearDataset = () => {
    setActiveDatasetId(null);
    setDatasetMetadata(null);
  };

  return (
    <DatasetContext.Provider
      value={{
        activeDatasetId,
        datasetMetadata,
        setActiveDataset,
        clearDataset,
      }}
    >
      {children}
    </DatasetContext.Provider>
  );
}

export function useDataset() {
  const context = useContext(DatasetContext);
  if (context === undefined) {
    throw new Error("useDataset must be used within a DatasetProvider");
  }
  return context;
}

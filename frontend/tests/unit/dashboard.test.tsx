import React from "react";
import { render, screen } from "@testing-library/react";
import { DatasetProvider } from "../../src/context/DatasetContext";
import Dashboard from "../../src/app/page";

describe("Dashboard Page Rendering", () => {
  it("renders the welcome hero title and action card links", () => {
    render(
      <DatasetProvider>
        <Dashboard />
      </DatasetProvider>
    );
    
    // Check main title
    const titleElement = screen.getByRole("heading", { name: /DataMine AI Classifier/i });
    expect(titleElement).toBeInTheDocument();
    
    // Check that card headers render
    expect(screen.getByText(/1\. Upload ARFF Dataset/i)).toBeInTheDocument();
    expect(screen.getByText(/2\. Inspect & Preprocess/i)).toBeInTheDocument();
    expect(screen.getByText(/3\. Train & Classify/i)).toBeInTheDocument();
    expect(screen.getByText(/4\. Compare Performance/i)).toBeInTheDocument();
  });
});

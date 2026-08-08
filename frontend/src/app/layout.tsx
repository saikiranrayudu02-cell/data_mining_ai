import type { Metadata } from "next";
import Navbar from "@/components/Navbar/Navbar";
import Sidebar from "@/components/Sidebar/Sidebar";
import { DatasetProvider } from "@/context/DatasetContext";
import { ToastProvider } from "@/context/ToastContext";
import "../styles/globals.css";

export const metadata: Metadata = {
  title: "DataMine AI Classifier",
  description: "Upload .arff datasets, train classifications, visualize decision trees, extract rules, and compare ML algorithms.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <DatasetProvider>
          <ToastProvider>
            <div className="app-layout">
              <Navbar />
              <div className="app-body">
                <Sidebar />
                <main className="app-content">
                  {children}
                </main>
              </div>
            </div>
          </ToastProvider>
        </DatasetProvider>

        {/* Global structures styling */}
        <style>{`
          .app-layout {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            background-color: var(--bg-primary);
            background-image: 
              radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.04) 0px, transparent 50%),
              radial-gradient(at 100% 100%, rgba(168, 85, 247, 0.04) 0px, transparent 50%);
          }
          
          .app-body {
            display: flex;
            flex: 1;
          }
          
          .app-content {
            flex: 1;
            padding: 2rem 2.5rem;
            overflow-y: auto;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
          }
        `}</style>
      </body>
    </html>
  );
}

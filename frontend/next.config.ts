import type { NextConfig } from "next";

// The backend URL is injected via environment variable.
// - Local dev:  set in .env.local  → http://127.0.0.1:8000
// - Production: set in Vercel dashboard → https://data-mining-ai-backend.onrender.com
const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Reverse-proxy /api/* and /plots/* to the FastAPI backend.
  // This lets the frontend call /api/v1/... without CORS issues and without
  // hardcoding the backend URL into the browser bundle.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
      {
        source: "/plots/:path*",
        destination: `${BACKEND_URL}/plots/:path*`,
      },
    ];
  },
};

export default nextConfig;

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Configure redirects/rewrites if we want to reverse-proxy backend API
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
      {
        source: "/plots/:path*",
        destination: "http://127.0.0.1:8000/plots/:path*",
      },
    ];
  },
};

export default nextConfig;

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is proxied so the browser only ever talks to one origin. In the sandbox
// preview the app is served from a proxied host, so an absolute backend URL would
// break: relative /api requests are forwarded here instead.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: false,
    allowedHosts: true,
    proxy: {
      "/api": { target: process.env.AE_API_URL ?? "http://127.0.0.1:8000", changeOrigin: true },
      "/healthz": { target: process.env.AE_API_URL ?? "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});

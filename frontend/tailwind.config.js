/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 900: "#0b0f14", 800: "#111823", 700: "#1a2331", 600: "#25303f" },
        accent: { DEFAULT: "#7c5cff", soft: "#a78bfa" },
        ok: "#34d399", warn: "#fbbf24", bad: "#f87171",
      },
      fontFamily: { mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"] },
    },
  },
  plugins: [],
};

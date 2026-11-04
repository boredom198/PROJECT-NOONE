/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#080B11",
        panel: "#0D131F",
        panel2: "#131B2C",
        line: "#1E293B",
        lineSubtle: "rgba(255, 255, 255, 0.07)",
        txt: "#F1F5F9",
        muted: "#94A3B8",
        faint: "#64748B",
        gain: "#10B981",
        gainMuted: "rgba(16, 185, 129, 0.15)",
        loss: "#F43F5E",
        lossMuted: "rgba(244, 63, 94, 0.15)",
        accent: "#3B82F6",
        accentMuted: "rgba(59, 130, 246, 0.15)",
        cyanAccent: "#06B6D4",
        warn: "#F59E0B",
        warnMuted: "rgba(245, 158, 11, 0.15)",
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "-apple-system", "sans-serif"],
        mono: ["Cascadia Mono", "Consolas", "ui-monospace", "monospace"],
      },
      borderRadius: {
        sm2: "6px",
        card: "10px",
      },
      boxShadow: {
        glass: "inset 0 1px 0 0 rgba(255, 255, 255, 0.05), 0 8px 24px -4px rgba(0, 0, 0, 0.4)",
        glow: "0 0 20px rgba(59, 130, 246, 0.15)",
        glowGain: "0 0 16px rgba(16, 185, 129, 0.2)",
      },
    },
  },
  plugins: [],
};

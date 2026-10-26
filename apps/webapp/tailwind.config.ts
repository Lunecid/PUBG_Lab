import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper:    "#fefcf7",
        cream:    "#f3eddd",
        ink:      "#1a1a1a",
        muted:    "#6b6558",
        faint:    "#cdc5b3",
        rule:     "#d8d1c2",
        accent:   "#b02a2a",
        accent2:  "#3a4a5a",
        accent3:  "#6b8e3a",
        gridline: "#ebe4d3",
        good:     "#15803d",
        warn:     "#d97706",
        bad:      "#b91c1c",
      },
      fontFamily: {
        sans: [
          "Inter",
          "Noto Sans KR",
          "ui-sans-serif",
          "system-ui",
          "sans-serif",
        ],
        serif: [
          "Lora",
          "Noto Serif KR",
          "ui-serif",
          "Georgia",
          "serif",
        ],
        news: [
          "Newsreader",
          "Noto Serif KR",
          "ui-serif",
          "Georgia",
          "serif",
        ],
        mono: [
          "IBM Plex Mono",
          "SF Mono",
          "Menlo",
          "Consolas",
          "monospace",
        ],
        jet: [
          "JetBrains Mono",
          "IBM Plex Mono",
          "SF Mono",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      maxWidth: {
        prose: "72ch",
      },
    },
  },
  plugins: [],
} satisfies Config;

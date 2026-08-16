/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,ts}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      fontFamily: {
        display: ["'General Sans'", "sans-serif"],
        sans: ["'Plus Jakarta Sans'", "sans-serif"]
      },
      colors: {
        ink: "rgb(var(--color-ink) / <alpha-value>)",
        secondary: "rgb(var(--color-secondary) / <alpha-value>)",
        muted: "rgb(var(--color-muted) / <alpha-value>)",
        placeholder: "rgb(var(--color-placeholder) / <alpha-value>)",
        canvas: "rgb(var(--color-canvas) / <alpha-value>)",
        surface: "rgb(var(--color-surface) / <alpha-value>)",
        sidebar: "rgb(var(--color-sidebar) / <alpha-value>)",
        hover: "rgb(var(--color-hover) / <alpha-value>)",
        line: "rgb(var(--color-line) / <alpha-value>)",
        accent: "rgb(var(--color-accent) / <alpha-value>)",
        accentHover: "rgb(var(--color-accent-hover) / <alpha-value>)",
        success: "rgb(var(--color-success) / <alpha-value>)",
        warning: "rgb(var(--color-warning) / <alpha-value>)",
        error: "rgb(var(--color-error) / <alpha-value>)",
        userBubble: "rgb(var(--color-user-bubble) / <alpha-value>)",
        userBubbleBorder: "rgb(var(--color-user-bubble-border) / <alpha-value>)",
        userBubbleText: "rgb(var(--color-user-bubble-text) / <alpha-value>)"
      },
      boxShadow: {
        glow: "0 4px 20px rgba(127, 22, 53, 0.08)",
        glass: "0 4px 20px rgba(127, 22, 53, 0.05), 0 1px 3px rgba(127, 22, 53, 0.04)",
        glassHover: "0 8px 30px rgba(127, 22, 53, 0.08), 0 2px 6px rgba(127, 22, 53, 0.06)"
      },
      backgroundImage: {
        aurora: "radial-gradient(circle at top left, rgba(243, 214, 223, 0.45), transparent 34%), radial-gradient(circle at top right, rgba(255, 240, 245, 0.6), transparent 36%), linear-gradient(135deg, rgba(255,255,255,0.9), rgba(252,231,238,0.7))"
      },
      animation: {
        float: "float 8s ease-in-out infinite",
        pulseGlow: "pulseGlow 4s ease-in-out infinite",
        "fade-in": "fadeIn 0.4s ease-out forwards",
        "fade-up": "fadeUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "fade-up-spring": "fadeUpSpring 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards",
        "fade-up-smooth": "fadeUpSmooth 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "slide-in-right": "slideInRight 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "accordion-down": "accordionDown 0.2s ease-out",
        "accordion-up": "accordionUp 0.2s ease-out",
        marquee: "marquee 40s linear infinite",
        blob: "blob 10s infinite"
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" }
        },
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" }
        },
        fadeUpSpring: {
          "0%": { opacity: "0", transform: "translateY(24px) scale(0.95)" },
          "100%": { opacity: "1", transform: "translateY(0) scale(1)" }
        },
        fadeUpSmooth: {
          "0%": { opacity: "0", transform: "translateY(12px) scale(0.98)" },
          "100%": { opacity: "1", transform: "translateY(0) scale(1)" }
        },
        slideInRight: {
          "0%": { opacity: "0", transform: "translateX(20px)" },
          "100%": { opacity: "1", transform: "translateX(0)" }
        },
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-12px)" }
        },
        pulseGlow: {
          "0%, 100%": { boxShadow: "0 0 0 rgba(124, 228, 255, 0.2)" },
          "50%": { boxShadow: "0 0 42px rgba(255, 123, 172, 0.3)" }
        },
        accordionDown: {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" }
        },
        accordionUp: {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" }
        },
        marquee: {
          "0%": { transform: "translateX(0%)" },
          "100%": { transform: "translateX(-100%)" }
        },
        blob: {
          "0%": { transform: "translate(0px, 0px) scale(1)" },
          "33%": { transform: "translate(30px, -50px) scale(1.1)" },
          "66%": { transform: "translate(-20px, 20px) scale(0.9)" },
          "100%": { transform: "translate(0px, 0px) scale(1)" }
        }
      }
    }
  },
  plugins: []
};

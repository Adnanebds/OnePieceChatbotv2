module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Poppins", "Roboto", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["Bebas Neue", "Anime Ace", "sans-serif"],
      },
      colors: {
        blue: {
          100: "#DBEAFE",
          300: "#93C5FD",
          600: "#2563EB",
          900: "#1E3A8A",
        },
        yellow: {
          100: "#FEF3C7",
          300: "#FBBF24",
          400: "#F59E0B",
          500: "#D97706",
        },
      },
    },
  },
};
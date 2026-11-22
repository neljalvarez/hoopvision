/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        coal: { 950: '#0d0e10', 900: '#131417', 850: '#181a1e', 800: '#1e2126', 700: '#2a2e35', 600: '#3a3f48', 500: '#5a606b' },
        ember: { 300: '#ffb27a', 400: '#ff8f3d', 500: '#ff6b00', 600: '#e05a00', 700: '#b34700' },
        chalk: { 50: '#faf8f3', 100: '#f2efe7', 200: '#dcd8ce', 300: '#b9b5ab', 400: '#8e8a82' },
        make: '#34d399',
        miss: '#f87171',
        unsure: '#fbbf24',
      },
      fontWeight: { 600: '600', 700: '700', 800: '800' },
      fontFamily: {
        display: ['"Barlow Condensed"', 'Impact', 'sans-serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};

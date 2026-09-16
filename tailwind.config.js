/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./index.html", "./content/*.js"],
    theme: {
        extend: {
            fontFamily: {
                sans: ['-apple-system', 'BlinkMacSystemFont', '"Roboto"', 'Helvetica', 'Arial', 'sans-serif'],
                serif: ['Charter', '"Bitstream Charter"', 'Georgia', 'serif'],
                mono: ['Menlo', 'Monaco', 'Consolas', '"Liberation Mono"', '"Courier New"', 'monospace'],
            }
        }
    },
    plugins: [],
}

// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./TeacherSignup.html",
    "./StudentSignup.html",
    "./VerifyEmail.html",
    "./GérerDocument.html",
    "./Page_de_Chat.html",
    "./src/**/*.ts",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Roboto', 'Poppins', 'sans-serif'],
        poppins: ['Poppins', 'sans-serif'],
        roboto: ['Roboto', 'sans-serif'],
        'google-sans': ['Google Sans', 'Roboto', 'sans-serif'],
      },
      colors: {
        // --- Your Requested Colors for Gemini Style ---
        'chat-page-bg': '#FFFFFF',         // *** MODIFIED: Main page background to WHITE ***
        'chat-interactive': '#339DFF',     // Bright blue for interactive elements
        'chat-interactive-hover': '#2588e6',// Darker hover for bright blue

        // --- Additional Gemini-inspired Colors (can still be used for contrast/accents) ---
        'gemini-sidebar-bg': '#FFFFFF',       // Sidebar background (white)
        'gemini-text-primary': '#202124',     // Main dark text
        'gemini-text-secondary': '#5F6368',   // Secondary gray text
        'gemini-user-bubble-bg': '#d1e3ff',   // User message bubble background (light blue)
        'gemini-user-bubble-text': '#0b57d0', // User message bubble text
        'gemini-bot-bubble-bg': '#f0f4f9',    // Bot message bubble background (very light gray/blue)
        'gemini-bot-bubble-text': '#3c4043',    // Bot message bubble text
        'gemini-history-selected-bg': '#e8f0fe', // BG for selected history item
        'gemini-history-selected-text': '#1967d2',// Text for selected history item
        'gemini-input-bg': '#FFFFFF',         // Background for the input area in the footer (inner part)
        'gemini-input-border': '#dadce0',     // Border for the input area

        // --- Keep Your Original Colors for Other Pages ---
        'page-bg': '#F5FAFF', // Original page-bg, might be used by other pages
        'text-heading': '#1C2D5A',
        // ... (rest of your original colors remain unchanged) ...
        'text-link-forgot': '#A0AEC0',
        'signup-bg': '#e0f7fa',
        'signup-card-bg': '#ffffff',
        'signup-primary': '#00796b',
        'signup-primary-darker': '#005a4f',
        'signup-primary-focus-ring': 'rgba(0, 121, 107, 0.5)',
        'signup-input-text': '#424242',
        'signup-input-placeholder': '#bdbdbd',
        'signup-input-border': '#e0e0e0',
        'signup-input-bg': '#f8f9fa',
        'signup-error': '#d32f2f',
        'signup-error-focus-ring': 'rgba(211, 47, 47, 0.5)',
        'btn-primary-bg': '#457BFF',
        'btn-primary-bg-hover': '#3568E6',
        'btn-primary-focus-ring': 'rgba(69, 123, 255, 0.5)',
        'btn-secondary-border': '#E0E0E0',
        'btn-text-light': '#ffffff',
        'verify-bg': '#F5FCFF',
        'verify-button-bg': '#1A73E8',
        'manage-bg': '#F0FAFF',
        'manage-interactive': '#1A73E8',
      },
       borderRadius: {
         'xl': '0.75rem',
         '2xl': '1rem',
         'full': '9999px',
         'chat-bubble': '12px', // You can use rounded-xl or this specific one
         // Keep original if needed
         'chat-input': '28px',
         'sidebar-item': '8px',
         'action-button': '9999px',
         'card': '1rem',
         'input': '0.5rem',
         'button': '0.5rem',
       },
       boxShadow: {
         'md': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
         'lg': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
         'xl': '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
         'input-footer': '0 -2px 10px rgba(0, 0, 0, 0.07)',
         // Keep original if needed
         'sidebar-right': '1px 0 3px rgba(0, 0, 0, 0.07)',
         'bubble': '0 1px 2px 0 rgba(60,64,67,0.25), 0 2px 6px 2px rgba(60,64,67,0.1)',
       },
       transitionProperty: {
        'width': 'width',
        'margin': 'margin-left, margin-right',
        'max-height': 'max-height',
       }
    },
  },
  plugins: [
      require('@tailwindcss/forms'),
  ],
}
// Base URL for the backend API
// Can be overridden by environment variable REACT_APP_API_URL or VITE_API_URL depending on build tool
export const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000/api/v1';
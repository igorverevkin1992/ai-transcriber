import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: '0.0.0.0', // Важно для работы внутри контейнера/облака
    hmr: {
        // Исправляет проблему с Hot Module Replacement в облачных IDE
        clientPort: 443 
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  }
});
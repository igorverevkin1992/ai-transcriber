#!/usr/bin/env bash
# Start both backend (FastAPI) and frontend (Vite) servers.
# Backend: http://localhost:8000
# Frontend: http://localhost:3000 (proxies /api to backend)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install Python dependencies if needed
pip install -q -r requirements.txt 2>/dev/null

# Install Node dependencies if needed
if [ ! -d "node_modules" ]; then
  npm install
fi

echo "Starting ABTGS Backend on :8000 and Frontend on :3000..."

# Start backend in background
python main.py &
BACKEND_PID=$!

# Start frontend
npx vite --port 3000 &
FRONTEND_PID=$!

# Cleanup on exit
cleanup() {
  echo "Stopping servers..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

wait

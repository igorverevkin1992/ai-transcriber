#!/usr/bin/env bash
# Start both backend (FastAPI) and frontend (Vite) servers.
# Backend: http://localhost:8000
# Frontend: http://localhost:3000 (proxies /api to backend)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install Python dependencies
echo "[1/2] Installing Python dependencies..."
pip install -q -r requirements.txt 2>/dev/null || {
  echo "ERROR: Failed to install Python dependencies" >&2
  exit 1
}

# Install Node dependencies if needed
if [ ! -d "node_modules" ]; then
  echo "[2/2] Installing Node dependencies..."
  npm install || {
    echo "ERROR: Failed to install Node dependencies" >&2
    exit 1
  }
else
  echo "[2/2] Node dependencies already installed."
fi

echo ""
echo "Starting ABTGS..."
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo ""

# Start backend in background
python main.py &
BACKEND_PID=$!

# Wait briefly for backend to start
sleep 2

# Check if backend started
if ! kill -0 $BACKEND_PID 2>/dev/null; then
  echo "ERROR: Backend failed to start" >&2
  exit 1
fi

# Start frontend
npx vite --port 3000 &
FRONTEND_PID=$!

# Cleanup on exit
cleanup() {
  echo ""
  echo "Stopping servers..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

wait

#!/bin/bash
# Kill existing
pkill -f "uvicorn backend.main:app"
pkill -f "vite"

# Start Backend
echo "Starting Backend..."
source venv/bin/activate
nohup uvicorn backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend running at PID $BACKEND_PID"

# Start Frontend
echo "Starting Frontend..."
cd frontend
nohup npm run dev -- --host > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend running at PID $FRONTEND_PID"

echo "MOSS Started."

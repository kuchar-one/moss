#!/bin/bash

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=5173

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo -e "\n${RED}Shutting down MOSS...${NC}"
    if [ -n "$BACKEND_PID" ]; then
        kill -9 $BACKEND_PID 2>/dev/null
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill -9 $FRONTEND_PID 2>/dev/null
    fi
    # Aggressive Fallback
    pkill -9 -f "uvicorn backend.main:app"
    pkill -9 -f "vite"
    # Kill any lingering python processes related to optimization if spawned separately (though they are threads usually)
    # But just in case
    exit 0
}

# Trap Ctrl+C and exit
trap cleanup SIGINT SIGTERM

start_backend() {
    echo -e "${BLUE}Starting Backend...${NC}"
    source venv/bin/activate
    nohup uvicorn backend.main:app --host 0.0.0.0 --port $BACKEND_PORT > backend.log 2>&1 &
    BACKEND_PID=$!
    echo "Backend PID: $BACKEND_PID"
}

start_frontend() {
    echo -e "${BLUE}Starting Frontend...${NC}"
    cd frontend
    nohup npm run dev -- --host > frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    echo "Frontend PID: $FRONTEND_PID"
}

restart_services() {
    echo -e "${RED}Stopping services...${NC}"
    if [ -n "$BACKEND_PID" ]; then kill $BACKEND_PID 2>/dev/null; fi
    if [ -n "$FRONTEND_PID" ]; then kill $FRONTEND_PID 2>/dev/null; fi
    # Ensure they are dead
    pkill -f "uvicorn backend.main:app"
    pkill -f "vite"
    
    echo -e "${GREEN}Restarting...${NC}"
    start_backend
    start_frontend
}

# Initial Start
start_backend
start_frontend

echo -e "\n${GREEN}MOSS is running!${NC}"
echo -e "Backend: http://localhost:$BACKEND_PORT"
echo -e "Frontend: http://localhost:$FRONTEND_PORT"

while true; do
    echo -e "\n${BLUE}Controls:${NC}"
    echo " [r] Restart services"
    echo " [q] Quit and shutdown"
    read -p "Select option: " -n 1 -r key
    echo "" 

    case $key in
        r|R)
            restart_services
            ;;
        q|Q)
            cleanup
            ;;
        *)
            ;;
    esac
done

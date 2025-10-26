#!/bin/bash
cd /Users/madanat/Documents/Grocery-agent/CalHacks-Agents
source ../.venv/bin/activate
export PYTHONPATH=/Users/madanat/Documents/Grocery-agent:$PYTHONPATH
echo "🚀 Starting Backend API on http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

#!/bin/bash
cd /Users/madanat/Documents/Grocery-agent
source .venv-worker/bin/activate
echo "🎙️  Starting Voice Worker (LiveKit + Claude)"
python Cooking-Companion/cooking_companion.py dev

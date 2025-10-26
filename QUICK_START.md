# 🚀 Quick Start Guide

## ✅ CURRENT STATUS

- **Backend API**: ✅ RUNNING on http://localhost:8000
- **Voice Worker**: Ready to start
- **iOS App**: Ready to run

---

## 📱 TO RUN THE APP

### 1️⃣ Backend (Already running!)

The backend is already running! You can:
- View API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Stop it: `pkill -f uvicorn` (if needed)

### 2️⃣ Voice Worker (Optional)

In a NEW terminal:
```bash
cd /Users/madanat/Documents/Grocery-agent
./start_voice_worker.sh
```

### 3️⃣ iOS App

Open Xcode:
```bash
open "Grocery-Agent-ios/Grocery Agent.xcodeproj"
```

Then:
- Press ⌘B to build
- Press ⌘R to run
- Choose a simulator or device

---

## 🧪 TEST THE BACKEND

Register a user:
```bash
curl -X POST http://localhost:8000/auth/register \
  -H 'Content-Type: application/json' \
  -d '{
    "email":"test@example.com",
    "username":"testuser",
    "password":"testpass123",
    "name":"Test User",
    "daily_calories":2200,
    "dietary_restrictions":["vegetarian"],
    "likes":["indian"]
  }'
```

Login (get token):
```bash
curl -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{
    "email":"test@example.com",
    "password":"testpass123"
  }'
```

---

## 🛑 TO STOP EVERYTHING

```bash
# Stop backend
pkill -f uvicorn

# Stop voice worker  
pkill -f cooking_companion
```

---

## 📝 WHAT'S RUNNING

- **Backend**: FastAPI with recipe/grocery agents
- **Voice Worker**: LiveKit + Claude for cooking instructions  
- **iOS App**: Your SwiftUI grocery management app

See `STARTUP_GUIDE.md` for full details!

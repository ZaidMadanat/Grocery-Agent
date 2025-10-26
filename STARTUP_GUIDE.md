# 🚀 Complete Startup Guide for Grocery Agent

## ✅ What's Running

- **Backend API**: `CalHacks-Agents` (FastAPI) 
- **Voice Worker**: `Cooking-Companion` (LiveKit + Claude)
- **iOS App**: `Grocery-Agent-ios`

## 📋 Quick Start

### Terminal 1: Backend API (FastAPI)
```bash
cd /Users/madanat/Documents/Grocery-agent
./start_backend.sh
```

**Or manually:**
```bash
cd CalHacks-Agents
source ../.venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Terminal 2: Voice Worker (LiveKit)
```bash
cd /Users/madanat/Documents/Grocery-agent
./start_voice_worker.sh
```

**Or manually:**
```bash
source .venv-worker/bin/activate
python Cooking-Companion/cooking_companion.py dev
```

### Terminal 3: iOS App
Open in Xcode:
```bash
open "Grocery-Agent-ios/Grocery Agent.xcodeproj"
```

Then build and run in the simulator or device.

## 🧪 Test the Backend

Once backend is running, test with:

### Health Check
```bash
curl http://localhost:8000/health
```

### Register a User
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
    "likes":["indian","italian"]
  }'
```

### Login (Get Token)
```bash
curl -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{
    "email":"test@example.com",
    "password":"testpass123"
  }'
```

### Generate Weekly Meals (replace TOKEN)
```bash
curl -X POST http://localhost:8000/weekly-meals/generate \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{"days": 7}'
```

### Generate Grocery List from Recipe
```bash
curl -X POST http://localhost:8000/grocery/from-recipe \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "title":"Paneer Tikka Bowl",
    "ingredients":[
      {"name":"paneer","quantity":200,"unit":"g"},
      {"name":"tomatoes","quantity":2,"unit":"piece"}
    ]
  }'
```

### Get LiveKit Session Token for Voice
```bash
curl -X POST http://localhost:8000/session/create \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{"room":"cooking-demo","ttl_seconds":3600}'
```

## 📱 iOS App Integration

### Current Status
Your iOS app already has **MockDataService** for all features. To connect to the real backend:

### What to Update

1. **API Configuration**: Add backend URL to your app:
   ```swift
   // In your AppViewModel or wherever you configure API
   private let baseURL = "http://localhost:8000"  // or your server URL
   ```

2. **Replace MockDataService**: Update your services to call real API:
   - `DashboardViewModel.swift` - Weekly meals API
   - `GroceryListViewModel.swift` - Grocery list API  
   - `InventoryViewModel.swift` - Inventory API
   - `NotificationsViewModel.swift` - Notifications API

3. **Authentication**: Store and use the JWT token from login:
   ```swift
   // After successful login
   UserDefaults.standard.set(token, forKey: "authToken")
   
   // In API calls
   var headers = ["Authorization": "Bearer \(token)"]
   ```

4. **Voice Integration**: To connect LiveKit voice:
   - Install LiveKit iOS SDK (add via SPM)
   - Call `/session/create` endpoint
   - Use returned `url`, `room`, and `token` to join

## 🔧 API Endpoints Available

- `GET /health` - Health check
- `GET /docs` - API documentation
- `POST /auth/register` - Register user
- `POST /auth/login` - Login (returns token)
- `POST /weekly-meals/generate` - Generate meal plan
- `POST /grocery/from-recipe` - Create grocery list
- `POST /session/create` - Get LiveKit token
- Inventory, notifications, and more...

## 🐛 Troubleshooting

### Backend won't start?
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Or use different port
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Voice worker errors?
```bash
# Reinstall voice worker deps
source .venv-worker/bin/activate
pip install -r Cooking-Companion/requirements.txt
```

### iOS can't connect?
- Make sure backend is running at `localhost:8000`
- For simulator: `localhost` works
- For physical device: Use your Mac's IP address (e.g., `192.168.1.x:8000`)

## 📖 Next Steps

1. ✅ Start backend in Terminal 1
2. ✅ Start voice worker in Terminal 2  
3. ✅ Open iOS app in Xcode
4. ✅ Test API endpoints
5. ✅ Replace mock data with real API calls
6. ✅ Add LiveKit SDK to iOS for voice features

## 🎯 What to Do NOW

1. **Open 3 terminals** or terminal tabs
2. **Terminal 1**: Run `./start_backend.sh`
3. **Terminal 2**: Run `./start_voice_worker.sh`
4. **Terminal 3**: Open Xcode with `open "Grocery-Agent-ios/Grocery Agent.xcodeproj"`
5. Build and run the iOS app
6. Test the app!

Your backend should be at: http://localhost:8000/docs (FastAPI docs)

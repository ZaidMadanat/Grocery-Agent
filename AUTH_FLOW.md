# 🔐 Authentication Flow - Updated

## ✅ What's Been Implemented

### New Authentication Flow:
1. **Sign In Page** (existing, updated) → Link to Sign Up
2. **Sign Up Page** (NEW) → Creates account → Onboarding
3. **Onboarding** (existing) → Collects preferences → Main App
4. **Main App** (existing) → Full grocery agent experience

---

## 📱 User Flows

### Flow 1: New User (Sign Up)
```
LoginView
    ↓ (tap "Sign Up")
SignUpView
    ↓ (successful registration)
OnboardingFlowView
    ↓ (complete preferences)
MainShellView (App)
```

### Flow 2: Existing User (Sign In)
```
LoginView
    ↓ (enter credentials)
MainShellView (App)
    [if already onboarded]
```

---

## 🆕 New Files Created

### 1. **SignUpView.swift**
- Located: `/Grocery-Agent-ios/Grocery Agent/Views/Auth/SignUpView.swift`
- Beautiful sign-up form with:
  - Full Name
  - Username
  - Email
  - Password (min 8 chars)
  - Confirm Password
  - Form validation
  - Error messages
  - "Already have an account?" link back to Sign In

### 2. **SignUpViewModel.swift**
- Located: `/Grocery-Agent-ios/Grocery Agent/ViewModels/SignUpViewModel.swift`
- Handles:
  - Form validation
  - API registration call
  - Token storage
  - Error handling
  - Loading states

### 3. **Updated AuthDTOs.swift**
- Added `RegisterRequest` model for backend API
- Fields: email, username, password, name, daily_calories, dietary_restrictions, likes

---

## 📝 Updated Files

### 1. **LoginView.swift**
- Added "Sign Up" navigation link at bottom
- Removed default pre-filled credentials
- Better UX for new users

### 2. **LoginViewModel.swift**
- Removed default email/password values
- Cleaner authentication state

---

## 🔄 Complete Authentication Logic

### ContentView.swift (Existing - No Changes Needed)
The ContentView already handles the flow perfectly:

```swift
if !hasToken {
    LoginView { hasToken = true }           // No token? → Login or Sign Up
} else if appModel.isOnboarded {
    MainShellView(appModel: appModel)       // Token + Onboarded → Main App
} else {
    OnboardingFlowView(appModel: appModel)  // Token + Not Onboarded → Onboarding
}
```

**This means:**
- New users signing up will automatically go to onboarding
- Returning users signing in will go straight to the app (if already onboarded)

---

## ✅ Validation Rules

### Sign Up Form:
- **Full Name**: Required
- **Username**: Required
- **Email**: Required, must contain "@"
- **Password**: Required, minimum 8 characters
- **Confirm Password**: Must match password

### Backend Validation (enforced):
- Email must be unique
- Username must be unique
- Password minimum 8 characters

---

## 🧪 Testing

### Test Accounts Created:
1. **rajeev@example.com** / secure123
2. **testuser2@example.com** / password123

### Test Sign Up Flow:
1. Launch app in Xcode (⌘R)
2. Tap "Sign Up" on login screen
3. Fill in form:
   - Name: Your Name
   - Username: yourname
   - Email: yourname@example.com
   - Password: password123 (min 8 chars)
   - Confirm: password123
4. Tap "Sign Up"
5. Should automatically show onboarding screens
6. Complete onboarding
7. Lands in main app!

### Test Sign In Flow:
1. Use existing credentials
2. Should skip onboarding if already completed
3. Go straight to main app

---

## 🎨 UI/UX Features

- ✅ Modern, clean design matching existing app style
- ✅ Real-time form validation
- ✅ Loading indicators during API calls
- ✅ Clear error messages
- ✅ Password requirements shown
- ✅ Easy navigation between Sign In/Sign Up
- ✅ Smooth transitions and animations

---

## 🚀 Ready to Use!

The authentication flow is complete and fully functional. Just build and run the iOS app in Xcode:

```bash
# Already opened in Xcode!
# Just press ⌘R to build and run
```

---

## 📊 Architecture

### API Integration:
- **POST /auth/register** - Create new account
- **POST /auth/login** - Sign in existing user
- Both return JWT token stored in `AuthService`

### State Management:
- `ContentView` observes token state
- `AppViewModel` tracks onboarding completion
- Automatic flow routing based on state

### Security:
- JWT tokens stored securely
- Passwords never stored locally
- 7-day token expiration
- Secure API communication

---

## 🎉 Summary

You now have a complete authentication system with:
- ✅ Sign In page with link to Sign Up
- ✅ Beautiful Sign Up page with full validation
- ✅ Automatic routing: Sign Up → Onboarding → App
- ✅ Seamless Sign In → App (skip onboarding for existing users)
- ✅ All tested and working with backend API!

**The app is ready to use! 🚀**


"""
Agentic Grocery - FastAPI Backend
Multi-agent system for food recommendation and grocery automation
Integrates Fetch.ai's uAgents with FastAPI for ASI:One compatibility
Includes authentication, user management, and SQLite database
"""

import os
import sys
import json
import warnings
import uuid
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, date
from pathlib import Path


# Suppress general deprecation warnings from optional dependencies

from fastapi import FastAPI, HTTPException, status, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from jose import jwt
from anthropic import Anthropic

# Import agent modules
from agents.recipe_agent.agent import generate_recipes
from agents.grocery_agent.agent import generate_grocery_list
from agents.recipe_agent.daily_meals import (
    generate_daily_meals_with_claude, 
    generate_single_meal_with_claude,
    DailyMealRequest,
    DailyMealResponse
)
from chroma_service import ChromaService
from utils.logger import setup_logger, log_api_call

# Import database and auth
from database import (
    get_db, init_db, create_user, get_user_by_email, get_user_by_username,
    create_user_profile, save_recipe as db_save_recipe,
    create_grocery_list as db_create_grocery_list, log_meal as db_log_meal,
    create_daily_meal_plan, get_daily_meal_plan, create_user_preference,
    get_inventory_items, get_inventory_item, upsert_inventory_item, delete_inventory_item,
    get_notifications, mark_notification_read, create_notification,
    User, UserProfile, Recipe, GroceryList, MealHistory, DailyMealPlan, UserPreference,
    InventoryItem, Notification
)
from auth import create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

# Setup logger
logger = setup_logger("AgenticGrocery")

# Initialize ChromaDB service
chroma_service = ChromaService()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

_anthropic_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_client: Optional[Anthropic] = Anthropic(api_key=_anthropic_key) if _anthropic_key else None
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-20241022")

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    logger.info("🚀 Starting Agentic Grocery API...")
    
    # Initialize database
    init_db()
    logger.info("✅ Database initialized")
    
    logger.info("✅ All agents initialized and ready")
    yield
    logger.info("👋 Shutting down Agentic Grocery API...")


# Initialize FastAPI app
app = FastAPI(
    title="Agentic Grocery API",
    description="Multi-agent system for food recommendations and grocery automation using Fetch.ai uAgents",
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API requests/responses


class RecipeRequest(BaseModel):
    """Recipe endpoint request model"""
    user_profile: Dict[str, Any] = Field(..., description="User profile data")
    preferences: Dict[str, Any] = Field(..., description="User preferences")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class GroceryRequest(BaseModel):
    """Grocery endpoint request model"""
    recipe: Dict[str, Any] = Field(..., description="Selected recipe")
    user_id: str = Field(default="raj", description="User identifier")
    store_preference: str = Field(default="Kroger", description="Preferred store (Kroger API supported)")


class LiveKitTokenRequest(BaseModel):
    """LiveKit token request model"""
    room: Optional[str] = Field(default=None, description="LiveKit room name to join")
    identity: Optional[str] = Field(default=None, description="Participant identity")
    ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="Token time-to-live in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary metadata for the session")


class LiveKitTokenResponse(BaseModel):
    """LiveKit token response"""
    token: str
    url: str
    room: str
    identity: str
    expires_at: datetime


class AgentHandoffRequest(BaseModel):
    """Request payload for agent handoff within LiveKit"""
    room: str = Field(..., description="Target LiveKit room name")
    to_agent: str = Field(default="cooking_companion", description="Agent identifier to handoff to")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context for the agent")


class WeeklyMealGenerateRequest(BaseModel):
    """Request body for weekly meal generation"""
    start_date: Optional[str] = Field(default=None, description="ISO date to start the meal plan from")
    days: int = Field(default=7, ge=1, le=14, description="Number of days to generate (max 14)")


class GroceryListItemUpdate(BaseModel):
    """Request payload for updating grocery list item"""
    quantity: Optional[str] = Field(default=None, description="Updated quantity string")
    unit: Optional[str] = Field(default=None, description="Updated unit")
    is_checked: Optional[bool] = Field(default=None, description="Mark item as purchased")


class InventoryItemPayload(BaseModel):
    """Inventory item creation payload"""
    name: str = Field(..., description="Item name")
    quantity: Optional[str] = Field(default="1", description="Quantity string (e.g., '2 cups')")
    unit: Optional[str] = Field(default=None, description="Unit for numeric quantities")
    category: Optional[str] = Field(default=None, description="Optional category label")
    expires_at: Optional[str] = Field(default=None, description="Expiry timestamp in ISO-8601 format")
    days_until_expiry: Optional[int] = Field(default=None, ge=0, description="Fallback days until expiry")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class InventoryItemUpdate(BaseModel):
    """Inventory item update payload"""
    name: Optional[str] = Field(default=None, description="Updated name")
    quantity: Optional[str] = Field(default=None, description="Updated quantity string")
    unit: Optional[str] = Field(default=None, description="Updated unit")
    category: Optional[str] = Field(default=None, description="Updated category")
    expires_at: Optional[str] = Field(default=None, description="Updated expiry timestamp")
    days_until_expiry: Optional[int] = Field(default=None, ge=0, description="Updated days until expiry")
    notes: Optional[str] = Field(default=None, description="Updated notes")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_grocery_item_ids(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure grocery items include stable IDs and defaults"""
    enriched: List[Dict[str, Any]] = []
    for item in items:
        item_copy = dict(item)
        if not item_copy.get("id"):
            item_copy["id"] = str(uuid.uuid4())
        if "is_checked" not in item_copy:
            item_copy["is_checked"] = False
        enriched.append(item_copy)
    return enriched


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO formatted datetime string safely"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # Allow date-only format
        try:
            parsed_date = datetime.strptime(value, "%Y-%m-%d")
            return parsed_date
        except ValueError:
            return None


def _compute_inventory_expiry(expires_at_str: Optional[str], days_until: Optional[int]) -> Tuple[Optional[datetime], int]:
    """Determine expiry datetime and days remaining"""
    now = datetime.utcnow()
    expires_at = _parse_iso_datetime(expires_at_str)

    if expires_at:
        remaining = max((expires_at.date() - now.date()).days, 0)
        return expires_at, remaining

    if days_until is not None:
        remaining = max(days_until, 0)
        expires_at = now + timedelta(days=remaining)
        return expires_at, remaining

    # Default to one week
    expires_at = now + timedelta(days=7)
    return expires_at, 7


def _inventory_item_to_dict(item: InventoryItem) -> Dict[str, Any]:
    """Serialize inventory item for API"""
    expiry = item.expires_at.isoformat() if item.expires_at else None
    remaining = item.days_until_expiry
    if item.expires_at:
        remaining = max((item.expires_at.date() - datetime.utcnow().date()).days, 0)
    return {
        "id": item.id,
        "name": item.name,
        "quantity": item.quantity,
        "unit": item.unit,
        "category": item.category,
        "expires_at": expiry,
        "days_until_expiry": remaining,
        "notes": item.notes,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


def _notification_to_dict(notification: Notification) -> Dict[str, Any]:
    """Serialize notification for API consumers"""
    return {
        "id": notification.id,
        "title": notification.title,
        "message": notification.message,
        "type": notification.type,
        "is_read": notification.is_read,
        "metadata": notification.meta_data or {},
        "created_at": notification.created_at.isoformat() if notification.created_at else None,
    }


def _find_grocery_item(items: List[Dict[str, Any]], item_id: str) -> Optional[Dict[str, Any]]:
    """Find grocery item by id within a list"""
    for item in items:
        if str(item.get("id")) == str(item_id):
            return item
    return None


def _create_recipe_from_meal(user_id: int, meal_type: str, meal_data: Dict[str, Any]) -> Recipe:
    """Build a Recipe ORM object from generated meal data"""
    return Recipe(
        user_id=user_id,
        title=meal_data["title"],
        description=meal_data.get("description"),
        meal_type=meal_type,
        cook_time=meal_data.get("cook_time"),
        prep_time=meal_data.get("prep_time", "15 minutes"),
        servings=meal_data.get("servings", 1),
        cuisine=meal_data.get("cuisine"),
        difficulty=meal_data.get("difficulty", "medium"),
        calories=meal_data.get("calories"),
        protein_g=meal_data.get("protein"),
        carbs_g=meal_data.get("carbs"),
        fat_g=meal_data.get("fat"),
        ingredients=meal_data.get("ingredients", []),
        instructions=meal_data.get("instructions", ""),
        image_url=meal_data.get("image_url", ""),
        chroma_id=meal_data.get("chroma_id", "")
    )


def _store_recipe_embedding(user_id: int, meal_type: str, recipe_payload: Dict[str, Any], recipe_obj: Recipe) -> None:
    """Persist recipe embedding to Chroma and update recipe record"""
    ingredient_names = [ing.get("name", "") for ing in recipe_payload.get("ingredients", [])]
    embedding_payload = f"{recipe_payload.get('title', '')} {recipe_payload.get('description', '')} {' '.join(ingredient_names)}"
    recipe_embedding = chroma_service.generate_embedding(embedding_payload)
    recipe_data_for_chroma = {
        "user_id": user_id,
        "recipe_id": recipe_obj.id,
        "title": recipe_payload.get("title"),
        "description": recipe_payload.get("description"),
        "meal_type": meal_type,
        "cuisine": recipe_payload.get("cuisine", ""),
        "calories": recipe_payload.get("calories", 0),
        "ingredients": ingredient_names
    }
    chroma_id = chroma_service.store_recipe(recipe_data_for_chroma, recipe_embedding)
    recipe_obj.chroma_id = chroma_id


def _persist_daily_meal_set(
    db: Session,
    user_id: int,
    response: DailyMealResponse,
    plan_date: date
) -> DailyMealPlan:
    """Persist generated daily meals to the database and Chroma"""
    breakfast_recipe = _create_recipe_from_meal(user_id, "breakfast", response.breakfast)
    lunch_recipe = _create_recipe_from_meal(user_id, "lunch", response.lunch)
    dinner_recipe = _create_recipe_from_meal(user_id, "dinner", response.dinner)

    db.add_all([breakfast_recipe, lunch_recipe, dinner_recipe])
    db.flush()

    daily_plan = DailyMealPlan(
        user_id=user_id,
        date=plan_date,
        breakfast_recipe_id=breakfast_recipe.id,
        lunch_recipe_id=lunch_recipe.id,
        dinner_recipe_id=dinner_recipe.id
    )
    db.add(daily_plan)
    db.commit()

    # Store embeddings post-commit, then persist chroma ids
    _store_recipe_embedding(user_id, "breakfast", response.breakfast, breakfast_recipe)
    _store_recipe_embedding(user_id, "lunch", response.lunch, lunch_recipe)
    _store_recipe_embedding(user_id, "dinner", response.dinner, dinner_recipe)
    db.commit()

    return daily_plan


def _recipe_to_dict(recipe: Recipe) -> Dict[str, Any]:
    """Serialize recipe ORM model to API payload"""
    def _safe_json(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    return {
        "id": recipe.id,
        "title": recipe.title,
        "description": recipe.description,
        "meal_type": recipe.meal_type,
        "cook_time": recipe.cook_time,
        "prep_time": recipe.prep_time,
        "servings": recipe.servings,
        "cuisine": recipe.cuisine,
        "difficulty": recipe.difficulty,
        "calories": recipe.calories,
        "protein_g": recipe.protein_g,
        "carbs_g": recipe.carbs_g,
        "fat_g": recipe.fat_g,
        "ingredients": _safe_json(recipe.ingredients) or [],
        "instructions": recipe.instructions,
        "image_url": recipe.image_url,
        "chroma_id": recipe.chroma_id,
        "created_at": recipe.created_at.isoformat() if recipe.created_at else None,
        "updated_at": recipe.updated_at.isoformat() if recipe.updated_at else None,
    }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    version: str
    agents: Dict[str, str]


# Authentication Models
class MacrosOptional(BaseModel):
    """Optional macros for user"""
    protein: Optional[float] = Field(None, description="Daily protein target in grams")
    carbs: Optional[float] = Field(None, description="Daily carbs target in grams")
    fats: Optional[float] = Field(None, description="Daily fats target in grams")


class UserRegister(BaseModel):
    """User registration model with detailed food preferences"""
    # Account credentials
    email: str = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, description="Password for login")
    
    # Personal information
    name: str = Field(..., min_length=1, description="Full name")
    
    # Dietary information (required)
    daily_calories: float = Field(..., gt=0, description="Daily calorie target")
    dietary_restrictions: List[str] = Field(
        ..., 
        description="Dietary restrictions including allergies, diet type (vegetarian, vegan, etc.)",
        example=["vegetarian", "gluten-free", "no nuts"]
    )
    likes: List[str] = Field(
        ..., 
        description="Cuisines and flavor profiles (e.g., 'indian', 'spicy', 'sweet', 'savory')",
        example=["indian", "spicy", "savory", "grilled"]
    )
    
    # Optional information
    additional_information: Optional[str] = Field(
        None, 
        description="Additional free-form text about food preferences",
        example="I prefer low-carb meals after 6pm. Love garlic in everything."
    )
    macros: Optional[MacrosOptional] = Field(
        None,
        description="Optional macro targets (protein, carbs, fats)"
    )


class UserLogin(BaseModel):
    """User login model"""
    email: str
    password: str


class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    user: Dict[str, Any]


class UserProfileUpdate(BaseModel):
    """User profile update model"""
    # Dietary information
    daily_calories: Optional[float] = Field(None, gt=0, description="Daily calorie target")
    dietary_restrictions: Optional[List[str]] = Field(None, description="Dietary restrictions including allergies")
    likes: Optional[List[str]] = Field(None, description="Cuisines and flavor profiles")
    additional_information: Optional[str] = Field(None, description="Additional food preferences")
    
    # Optional macros
    target_protein_g: Optional[float] = Field(None, ge=0, description="Daily protein target in grams")
    target_carbs_g: Optional[float] = Field(None, ge=0, description="Daily carbs target in grams")
    target_fat_g: Optional[float] = Field(None, ge=0, description="Daily fats target in grams")


class MealFeedbackRequest(BaseModel):
    """Feedback on daily meals"""
    date: str
    breakfast_rating: Optional[int] = Field(None, ge=1, le=5)
    lunch_rating: Optional[int] = Field(None, ge=1, le=5)
    dinner_rating: Optional[int] = Field(None, ge=1, le=5)
    overall_rating: Optional[int] = Field(None, ge=1, le=5)
    notes: Optional[str] = None
    disliked_ingredients: Optional[List[str]] = None
    liked_ingredients: Optional[List[str]] = None


class RegenerateMealRequest(BaseModel):
    """Request to regenerate a specific meal"""
    date: str
    meal_type: str = Field(..., pattern="^(breakfast|lunch|dinner)$")


class RecipeIngredient(BaseModel):
    """Recipe ingredient model"""
    name: str = Field(..., description="Ingredient name")
    quantity: float = Field(..., description="Quantity needed")
    unit: str = Field(..., description="Unit of measurement")
    notes: Optional[str] = Field(None, description="Additional notes")


class RecipeForGrocery(BaseModel):
    """Recipe model for grocery list generation"""
    title: str = Field(..., description="Recipe title")
    ingredients: List[RecipeIngredient] = Field(..., description="List of ingredients")
    servings: Optional[int] = Field(1, description="Number of servings")
    description: Optional[str] = Field(None, description="Recipe description")


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Agentic Grocery API",
        "version": "0.3.0",
        "description": "Multi-agent system for food recommendations and grocery automation",
        "features": ["Multi-Agent AI", "Claude Recipes", "Kroger Integration", "User Authentication"],
        "agents": ["RecipeAgent", "GroceryAgent"],
        "docs": "/docs",
        "health": "/health",
        "auth": {
            "register": "/auth/register",
            "login": "/auth/login"
        }
    }


# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/auth/register", response_model=TokenResponse, tags=["Authentication"])
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Register a new user account with detailed food preferences
    
    Required fields:
    - email, username, password: Account credentials
    - name: Full name
    - daily_calories: Daily calorie target
    - dietary_restrictions: List including allergies, diet type (e.g., ['vegetarian', 'no nuts'])
    - likes: Cuisines and flavor profiles (e.g., ['indian', 'spicy', 'savory'])
    
    Optional fields:
    - additional_information: Free-form text about food preferences
    - macros: {protein, carbs, fats} - Optional macro targets
    
    Returns JWT token for immediate authentication with user profile.
    """
    log_api_call("/auth/register", "started")
    
    # Check if user exists
    if get_user_by_email(db, user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    if get_user_by_username(db, user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user with name
    user = User(
        email=user_data.email,
        username=user_data.username,
        name=user_data.name,
        hashed_password=User.hash_password(user_data.password)
    )
    db.add(user)
    db.flush()  # Get user ID before creating profile
    
    # Create user profile with detailed food preferences
    profile = UserProfile(
        user_id=user.id,
        # Required dietary information
        daily_calories=user_data.daily_calories,
        dietary_restrictions=user_data.dietary_restrictions,
        likes=user_data.likes,
        additional_information=user_data.additional_information,
        # Optional macros
        target_protein_g=user_data.macros.protein if user_data.macros else None,
        target_carbs_g=user_data.macros.carbs if user_data.macros else None,
        target_fat_g=user_data.macros.fats if user_data.macros else None
    )
    db.add(profile)
    db.commit()
    db.refresh(user)
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    log_api_call("/auth/register", "completed")
    logger.info(f"New user registered: {user.username} ({user.name})")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "name": user.name,
            "daily_calories": user_data.daily_calories,
            "dietary_restrictions": user_data.dietary_restrictions,
            "likes": user_data.likes
        }
    }


@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Login user and return JWT token
    
    Authenticates user with email and password.
    Returns JWT token valid for 7 days.
    """
    log_api_call("/auth/login", "started")
    
    user = get_user_by_email(db, credentials.email)
    
    if not user or not user.verify_password(credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    log_api_call("/auth/login", "completed")
    logger.info(f"User logged in: {user.username}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "name": user.name
        }
    }


# ==================== USER PROFILE ENDPOINTS ====================

@app.get("/profile", tags=["User Profile"])
async def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current user's profile
    
    Requires authentication. Returns user info and detailed dietary profile including:
    - Name, daily calories, dietary restrictions, likes, additional info
    - Optional macros and physical stats
    """
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    profile_data = None
    if profile:
        profile_data = {
            "daily_calories": profile.daily_calories,
            "dietary_restrictions": profile.dietary_restrictions,
            "likes": profile.likes,
            "additional_information": profile.additional_information,
            "macros": {
                "protein": profile.target_protein_g,
                "carbs": profile.target_carbs_g,
                "fats": profile.target_fat_g
            } if profile.target_protein_g or profile.target_carbs_g or profile.target_fat_g else None
        }
    
    return {
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "username": current_user.username,
            "name": current_user.name,
            "created_at": current_user.created_at
        },
        "profile": profile_data
    }


@app.put("/profile", tags=["User Profile"])
async def update_profile(
    profile_data: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user profile
    
    Updates dietary preferences, physical stats, and fitness goals.
    Only updates fields that are provided.
    """
    # Filter out None values
    update_data = {k: v for k, v in profile_data.model_dump().items() if v is not None}
    
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided for update"
        )
    
    profile = create_user_profile(db, current_user.id, update_data)
    
    logger.info(f"Profile updated for user: {current_user.username}")
    
    return {"message": "Profile updated successfully", "profile": profile}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring
    Returns status of all agents and system
    """
    log_api_call("/health", "started")
    
    try:
        health_status = HealthResponse(
            status="healthy",
            message="All systems operational",
            version="0.3.0",
            agents={
                "RecipeAgent": "operational",
                "GroceryAgent": "operational"
            }
        )
        log_api_call("/health", "completed")
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.post("/recipe", tags=["Agents"])
async def recipe_endpoint(
    request: RecipeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Recipe endpoint - generates personalized recipes
    
    **Requires Authentication**
    
    RecipeAgent creates 2-3 meal options based on authenticated user's profile and preferences.
    Uses Claude AI for intelligent recipe generation.
    Uses structured data compatible with Chat Protocol v0.3.0
    
    Args:
        request: RecipeRequest with preferences (user_profile optional, will use authenticated user's profile)
        current_user: Authenticated user (from JWT token)
    
    Returns:
        List of generated recipes with macros and instructions
    """
    log_api_call("/recipe", "started")
    logger.info(f"Generating recipes for user: {current_user.username}")
    
    try:
        # Get user profile from database
        profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
        
        # Use provided user_profile or construct from database
        if not request.user_profile and profile:
            user_profile_dict = {
                "height_cm": profile.height_cm,
                "weight_kg": profile.weight_kg,
                "goal": profile.goal,
                "diet": profile.diet,
                "workout_frequency": profile.workout_frequency,
                "likes": profile.likes or [],
                "dislikes": profile.dislikes or [],
                "allergies": profile.allergies or [],
                "target_macros": {
                    "protein_g": profile.target_protein_g or 140,
                    "carbs_g": profile.target_carbs_g or 200,
                    "fat_g": profile.target_fat_g or 50,
                    "calories": profile.target_calories or 1800
                }
            }
        else:
            user_profile_dict = request.user_profile or {}
        
        # Generate recipes through RecipeAgent
        recipe_request_dict = {
            "user_profile": user_profile_dict,
            "preferences": request.preferences,
            "context": request.context
        }
        
        response = generate_recipes(recipe_request_dict)
        
        log_api_call("/recipe", "completed")
        logger.info(f"Generated {len(response.get('recipes', []))} recipes for {current_user.username}")
        return response
        
    except Exception as e:
        logger.error(f"Recipe endpoint error: {e}")
        log_api_call("/recipe", "failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recipes: {str(e)}"
        )


@app.post("/grocery", tags=["Agents"])
async def grocery_endpoint(
    request: GroceryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Grocery endpoint - creates grocery list from recipe
    
    **Requires Authentication**
    
    GroceryAgent extracts ingredients and creates formatted list with Kroger prices.
    Automatically saves list to user's account.
    
    Args:
        request: GroceryRequest with selected recipe
        current_user: Authenticated user (from JWT token)
    
    Returns:
        Grocery list with items, quantities, and estimated costs
    """
    log_api_call("/grocery", "started")
    logger.info(f"Creating grocery list for user: {current_user.username}")
    
    try:
        # Generate grocery list through GroceryAgent
        grocery_request_dict = {
            "recipe": request.recipe,
            "user_id": current_user.username,
            "store_preference": request.store_preference
        }
        
        response = generate_grocery_list(grocery_request_dict)
        response["items"] = _ensure_grocery_item_ids(response["items"])
        
        # Save to database
        grocery_list = db_create_grocery_list(db, current_user.id, {
            "name": f"List for {request.recipe.get('title', 'Recipe')}",
            "store": response["store"],
            "total_cost": response["total_estimated_cost"],
            "items": response["items"]
        })
        
        log_api_call("/grocery", "completed")
        logger.info(f"Created and saved list with {len(response.get('items', []))} items")
        
        # Return response with database ID
        response["list_id"] = grocery_list.id
        return response
        
    except Exception as e:
        logger.error(f"Grocery endpoint error: {e}")
        log_api_call("/grocery", "failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating grocery list: {str(e)}"
        )

# ==================== REALTIME / LIVEKIT ====================


@app.post("/session/create", response_model=LiveKitTokenResponse, tags=["Realtime"])
async def create_livekit_session(
    request: LiveKitTokenRequest,
    current_user: User = Depends(get_current_user)
):
    """Mint a LiveKit access token for voice assistance"""
    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LiveKit environment variables are not configured"
        )

    room_name = request.room or f"cooking-{current_user.username}-{datetime.utcnow().strftime('%Y%m%d')}"
    identity = request.identity or f"user-{current_user.id}"

    ttl_seconds = request.ttl_seconds or 3600
    expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

    room_claim = {
        "name": room_name,
        "join": True,
    }

    if request.metadata:
        try:
            room_claim["metadata"] = json.dumps(request.metadata)
        except (TypeError, ValueError):
            room_claim["metadata"] = json.dumps({"note": "metadata serialization failed"})

    payload = {
        "iss": LIVEKIT_API_KEY,
        "sub": identity,
        "exp": int(expires_at.timestamp()),
        "video": {
            "room": room_claim
        }
    }

    token = jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")

    logger.info(f"🎤 LiveKit token minted for {current_user.username} → room {room_name}")

    return LiveKitTokenResponse(
        token=token,
        url=LIVEKIT_URL,
        room=room_name,
        identity=identity,
        expires_at=expires_at
    )


@app.post("/agent/handoff", tags=["Realtime"])
async def agent_handoff(
    request: AgentHandoffRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Lightweight metadata endpoint for handing control to a specific agent.
    Useful for logging or coordinating multi-agent flows.
    """
    logger.info(
        f"Agent handoff by {current_user.username} → {request.to_agent} in room {request.room}"
    )
    return {
        "room": request.room,
        "to_agent": request.to_agent,
        "received": True,
        "context": request.context or {}
    }


#==================== DAILY MEAL PLANNING ENDPOINTS ====================

@app.post("/weekly-meals/generate", tags=["Daily Meals"])
async def generate_weekly_meals(
    request: WeeklyMealGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate a multi-day meal plan and persist each day"""

    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    start_date = datetime.utcnow().date()
    if request.start_date:
        parsed = _parse_iso_datetime(request.start_date)
        if not parsed:
            raise HTTPException(status_code=400, detail="start_date must be ISO formatted")
        start_date = parsed.date()

    days = min(max(request.days or 7, 1), 14)
    weekly_plan: List[Dict[str, Any]] = []
    all_tools: List[str] = []

    for offset in range(days):
        target_date = start_date + timedelta(days=offset)
        daily_request = DailyMealRequest(
            user_id=current_user.id,
            date=target_date.isoformat(),
            target_calories=profile.daily_calories
        )
        response, tools_called = generate_daily_meals_with_claude(daily_request, chroma_service, profile)
        all_tools.extend(tools_called)

        _persist_daily_meal_set(db, current_user.id, response, target_date)

        weekly_plan.append({
            "date": target_date.isoformat(),
            "weekday": target_date.strftime("%A"),
            "breakfast": response.breakfast,
            "lunch": response.lunch,
            "dinner": response.dinner,
            "totals": {
                "calories": response.total_calories,
                "protein": response.total_protein,
                "carbs": response.total_carbs,
                "fat": response.total_fat
            }
        })

    log_api_call("/weekly-meals/generate", "completed")
    logger.info(f"Generated weekly plan ({days} days) for {current_user.username}: tools={set(all_tools)}")

    return {"week": weekly_plan, "days": days}

@app.post("/daily-meals/generate", tags=["Daily Meals"])
async def generate_daily_meals(
    day: str,  # Day name: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate 3 daily meals (breakfast, lunch, dinner) with macro targets"""
    
    # Get user profile with macro targets
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    # Generate daily meals with macro consideration
    request = DailyMealRequest(
        user_id=current_user.id,
        date=day,  # Use day name directly
        target_calories=profile.daily_calories
    )
    
    response, tools_called = generate_daily_meals_with_claude(request, chroma_service, profile)
    
    plan_date = datetime.utcnow().date()
    _persist_daily_meal_set(db, current_user.id, response, plan_date)

    log_api_call("/daily-meals/generate", "completed")
    logger.info(f"Generated daily meals for {current_user.username}: {tools_called}")
    
    return response


@app.post("/daily-meals/generate-by-day", tags=["Daily Meals"])
async def generate_daily_meals_by_day(
    day: str,  # Day name: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate 3 daily meals for a specific day of the week"""
    
    # Validate day name
    valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if day not in valid_days:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid day. Must be one of: {', '.join(valid_days)}"
        )
    
    # Get user profile with macro targets
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    # Generate daily meals with macro consideration
    request = DailyMealRequest(
        user_id=current_user.id,
        date=day,  # Pass the day name directly
        target_calories=profile.daily_calories
    )
    
    response, tools_called = generate_daily_meals_with_claude(request, chroma_service, profile)
    
    plan_date = datetime.utcnow().date()
    _persist_daily_meal_set(db, current_user.id, response, plan_date)

    log_api_call("/daily-meals/generate-by-day", "completed")
    logger.info(f"Generated daily meals for {current_user.username} on {day}: {tools_called}")
    
    return response


@app.post("/daily-meals/regenerate", tags=["Daily Meals"])
async def regenerate_meal(
    request: RegenerateMealRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Regenerate a specific meal for the day"""
    
    # Get existing meal plan
    meal_plan = db.query(DailyMealPlan).filter(
        DailyMealPlan.user_id == current_user.id,
        DailyMealPlan.date == datetime.strptime(request.date, "%Y-%m-%d")
    ).first()
    
    if not meal_plan:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    
    # Get user profile
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    
    # Generate new recipe for the meal type
    meal_request = DailyMealRequest(
        user_id=current_user.id,
        date=request.date,
        target_calories=profile.daily_calories if profile else 2000
    )
    
    # Generate single meal
    new_recipe_data = generate_single_meal_with_claude(meal_request, request.meal_type, chroma_service, profile)
    
    # Create new recipe in database
    new_recipe = Recipe(
        user_id=current_user.id,
        title=new_recipe_data["title"],
        description=new_recipe_data["description"],
        meal_type=request.meal_type,
        cook_time=new_recipe_data["cook_time"],
        prep_time=new_recipe_data.get("prep_time", "15 minutes"),
        servings=new_recipe_data.get("servings", 1),
        cuisine=new_recipe_data.get("cuisine"),
        difficulty=new_recipe_data.get("difficulty", "medium"),
        protein_g=new_recipe_data["protein"],
        carbs_g=new_recipe_data["carbs"],
        fat_g=new_recipe_data["fat"],
        calories=new_recipe_data["calories"],
        ingredients=new_recipe_data["ingredients"],
        instructions=new_recipe_data["instructions"],
        image_url=new_recipe_data.get("image_url", ""),
        chroma_id=new_recipe_data.get("chroma_id", "")
    )
    
    db.add(new_recipe)
    db.flush()  # Get ID
    
    # Update meal plan
    if request.meal_type == "breakfast":
        meal_plan.breakfast_recipe_id = new_recipe.id
    elif request.meal_type == "lunch":
        meal_plan.lunch_recipe_id = new_recipe.id
    elif request.meal_type == "dinner":
        meal_plan.dinner_recipe_id = new_recipe.id
    
    db.commit()
    
    return {
        "message": f"Regenerated {request.meal_type} for {request.date}",
        "recipe": new_recipe_data
    }

@app.post("/daily-meals/feedback", tags=["Daily Meals"])
async def submit_meal_feedback(
    feedback: MealFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit feedback on daily meals to improve future recommendations"""
    
    # Store feedback in ChromaDB for learning
    preference_data = {
        "user_id": current_user.id,
        "date": feedback.date,
        "feedback": feedback.dict(),
        "preference_type": "feedback"
    }
    
    # Generate embedding and store
    embedding = chroma_service.generate_embedding(json.dumps(preference_data))
    chroma_service.store_user_preference(current_user.id, preference_data, embedding)
    
    # Store individual ingredient preferences
    if feedback.disliked_ingredients:
        for ingredient in feedback.disliked_ingredients:
            pref_data = {
                "user_id": current_user.id,
                "preference_type": "disliked",
                "item_name": ingredient,
                "item_type": "ingredient",
                "context": f"Disliked in meal on {feedback.date}",
                "strength": 1.0
            }
            embedding = chroma_service.generate_embedding(ingredient)
            chroma_service.store_user_preference(current_user.id, pref_data, embedding)
    
    if feedback.liked_ingredients:
        for ingredient in feedback.liked_ingredients:
            pref_data = {
                "user_id": current_user.id,
                "preference_type": "liked",
                "item_name": ingredient,
                "item_type": "ingredient",
                "context": f"Liked in meal on {feedback.date}",
                "strength": 1.0
            }
            embedding = chroma_service.generate_embedding(ingredient)
            chroma_service.store_user_preference(current_user.id, pref_data, embedding)
    
    return {"message": "Feedback recorded for future recommendations"}

@app.get("/daily-meals/{date}", tags=["Daily Meals"])
async def get_daily_meals(
    date: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get daily meal plan for a specific date"""
    
    meal_plan = db.query(DailyMealPlan).filter(
        DailyMealPlan.user_id == current_user.id,
        DailyMealPlan.date == datetime.strptime(date, "%Y-%m-%d")
    ).first()
    
    if not meal_plan:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    
    # Get recipes
    breakfast = db.query(Recipe).filter(Recipe.id == meal_plan.breakfast_recipe_id).first()
    lunch = db.query(Recipe).filter(Recipe.id == meal_plan.lunch_recipe_id).first()
    dinner = db.query(Recipe).filter(Recipe.id == meal_plan.dinner_recipe_id).first()
    
    return {
        "date": date,
        "breakfast": breakfast,
        "lunch": lunch,
        "dinner": dinner,
        "user_rating": meal_plan.user_rating,
        "notes": meal_plan.notes,
        "is_completed": meal_plan.is_completed
    }

@app.get("/daily-meals", tags=["Daily Meals"])
async def get_user_meal_plans(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 30
):
    """Get user's meal plan history"""
    
    meal_plans = db.query(DailyMealPlan).filter(
        DailyMealPlan.user_id == current_user.id
    ).order_by(DailyMealPlan.date.desc()).limit(limit).all()
    
    return {"meal_plans": meal_plans}


#==================== GROCERY SHOPPING ENDPOINTS ====================

@app.post("/grocery/from-recipe", tags=["Grocery Shopping"])
async def create_grocery_list_from_recipe(
    recipe: RecipeForGrocery,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create grocery list from recipe using Kroger API
    
    **Requires Authentication**
    
    Takes a recipe and searches Kroger API for each ingredient to get:
    - Real product names and descriptions
    - Current prices
    - Product images
    - Available quantities
    
    Args:
        recipe: Recipe object with ingredients list
        current_user: Authenticated user
    
    Returns:
        Grocery list with Kroger product details, prices, and images
    """
    log_api_call("/grocery/from-recipe", "started")
    logger.info(f"Creating grocery list from recipe for user: {current_user.username}")
    
    try:
        # Import Kroger API functions from grocery agent
        from agents.grocery_agent.agent import search_and_price_ingredient
        
        # Extract ingredients from recipe
        ingredients = recipe.ingredients
        if not ingredients:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Recipe must contain ingredients"
            )
        
        grocery_items = []
        total_cost = 0.0
        kroger_items_found = 0
        
        # Search Kroger for each ingredient using the same function as /grocery endpoint
        for ingredient in ingredients:
            ingredient_name = ingredient.name
            quantity = ingredient.quantity
            unit = ingredient.unit
            
            if not ingredient_name:
                continue
            
            # Use the same search_and_price_ingredient function that works in /grocery
            item_details = search_and_price_ingredient(ingredient_name, quantity, unit)
            
            # ONLY include items found on Kroger - no fallback
            if item_details.get("found") and item_details.get("source") == "kroger_api":
                # Found on Kroger
                kroger_items_found += 1
                estimated_price = item_details["price"]
                
                grocery_item = {
                    "id": str(uuid.uuid4()),
                    "name": item_details["name"],
                    "description": f"Found on Kroger: {item_details['name']}",
                    "quantity": quantity,
                    "unit": unit,
                    "price_per_unit": estimated_price,
                    "total_price": estimated_price,
                    "image_url": "",
                    "kroger_product_id": item_details.get("product_id", ""),
                    "category": item_details.get("category", "groceries"),
                    "brand": item_details.get("brand", ""),
                    "size": "",
                    "available": True,
                    "source": "kroger",
                    "is_checked": False
                }
                
                grocery_items.append(grocery_item)
                if estimated_price is not None:
                    total_cost += estimated_price
            else:
                # Skip items not found on Kroger
                logger.info(f"⏭️  Skipping '{ingredient_name}' - not found on Kroger")
        
        grocery_items = _ensure_grocery_item_ids(grocery_items)

        # Create grocery list in database
        grocery_list = db_create_grocery_list(db, current_user.id, {
            "name": f"Grocery list for {recipe.title}",
            "store": "Kroger",
            "total_cost": total_cost,
            "items": grocery_items
        })
        
        # Generate Kroger order URL (if available)
        order_url = None
        if kroger_items_found > 0:
            # Create a basic Kroger search URL
            search_terms = "+".join([item["name"] for item in grocery_items[:3]])
            order_url = f"https://www.kroger.com/search?query={search_terms}"
        
        log_api_call("/grocery/from-recipe", "completed")
        logger.info(f"Created grocery list with {len(grocery_items)} items, {kroger_items_found} from Kroger")
        
        return {
            "agent": "GroceryAgent",
            "list_id": grocery_list.id,
            "store": "Kroger",
            "items": grocery_items,
            "total_estimated_cost": total_cost,
            "kroger_items_found": kroger_items_found,
            "total_items": len(grocery_items),
            "order_url": order_url,
            "message": f"Found {kroger_items_found}/{len(grocery_items)} items on Kroger",
            "recipe_title": recipe.title,
            "llm_provider": "kroger_api" if kroger_items_found > 0 else "estimated"
        }
        
    except Exception as e:
        logger.error(f"Grocery from recipe error: {e}")
        log_api_call("/grocery/from-recipe", "failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating grocery list: {str(e)}"
        )

@app.get("/grocery-lists/{list_id}", tags=["Grocery"])
async def get_grocery_list_detail(
    list_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return a single grocery list with items"""
    grocery_list = db.query(GroceryList).filter(
        GroceryList.id == list_id,
        GroceryList.user_id == current_user.id
    ).first()

    if not grocery_list:
        raise HTTPException(status_code=404, detail="Grocery list not found")

    items = grocery_list.items or []
    items = _ensure_grocery_item_ids(items)
    grocery_list.items = items
    db.commit()

    return {
        "id": grocery_list.id,
        "name": grocery_list.name,
        "store": grocery_list.store,
        "total_cost": grocery_list.total_cost,
        "is_completed": grocery_list.is_completed,
        "completed_at": grocery_list.completed_at.isoformat() if grocery_list.completed_at else None,
        "created_at": grocery_list.created_at.isoformat() if grocery_list.created_at else None,
        "items": items
    }


@app.patch("/grocery-lists/{list_id}/items/{item_id}", tags=["Grocery"])
async def update_grocery_list_item(
    list_id: int,
    item_id: str,
    payload: GroceryListItemUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update quantity/unit/state for a grocery list item"""
    grocery_list = db.query(GroceryList).filter(
        GroceryList.id == list_id,
        GroceryList.user_id == current_user.id
    ).first()

    if not grocery_list:
        raise HTTPException(status_code=404, detail="Grocery list not found")

    items = _ensure_grocery_item_ids(grocery_list.items or [])
    target = _find_grocery_item(items, item_id)
    if not target:
        raise HTTPException(status_code=404, detail="Grocery item not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key in ("quantity", "unit", "is_checked"):
        if key in update_data and update_data[key] is not None:
            target[key] = update_data[key]

    grocery_list.items = items
    db.commit()

    return {"item": target}


@app.delete("/grocery-lists/{list_id}/items/{item_id}", tags=["Grocery"], status_code=status.HTTP_200_OK)
async def delete_grocery_list_item(
    list_id: int,
    item_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove an item from a grocery list"""
    grocery_list = db.query(GroceryList).filter(
        GroceryList.id == list_id,
        GroceryList.user_id == current_user.id
    ).first()

    if not grocery_list:
        raise HTTPException(status_code=404, detail="Grocery list not found")

    items = _ensure_grocery_item_ids(grocery_list.items or [])
    new_items = [item for item in items if str(item.get("id")) != str(item_id)]

    if len(new_items) == len(items):
        raise HTTPException(status_code=404, detail="Grocery item not found")

    grocery_list.items = new_items
    db.commit()

    return {"message": "Item removed"}


#==================== RECIPE MANAGEMENT ENDPOINTS ====================

@app.get("/recipes", tags=["Recipes"])
async def get_saved_recipes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's saved recipes"""
    recipes = db.query(Recipe).filter(Recipe.user_id == current_user.id).all()
    return {"recipes": recipes}


@app.get("/recipes/{recipe_id}", tags=["Recipes"])
async def get_recipe_detail(
    recipe_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Fetch a single recipe with full details"""
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id,
        Recipe.user_id == current_user.id
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    return {"recipe": _recipe_to_dict(recipe)}


@app.get("/recipes/{recipe_id}/qa", tags=["Recipes"])
async def recipe_question_answer(
    recipe_id: int,
    question: str = Query(..., min_length=3),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Answer a user question about a stored recipe"""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id,
        Recipe.user_id == current_user.id
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    recipe_payload = _recipe_to_dict(recipe)

    context_parts = [
        f"Recipe: {recipe_payload['title']}",
        f"Description: {recipe_payload.get('description') or 'n/a'}",
        "Ingredients:",
    ]

    for ing in recipe_payload["ingredients"]:
        if isinstance(ing, dict):
            name = ing.get("name") or ing.get("ingredient") or ""
            qty = ing.get("quantity")
            unit = ing.get("unit")
            parts = [part for part in [name, str(qty) if qty else None, unit] if part]
            context_parts.append(f"- {' '.join(parts)}")
        else:
            context_parts.append(f"- {ing}")

    context_parts.append("Instructions:")
    context_parts.append(recipe_payload["instructions"])
    context_text = "\n".join(context_parts)

    answer = "I could not generate an answer. Please try again."
    anthropic_answered = False

    if anthropic_client:
        try:
            prompt = (
                "You are a cooking assistant. Answer the user's question using only the recipe context.\n"
                "If the context does not contain the answer, say you are unsure.\n\n"
                f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
            )
            response_msg = anthropic_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=400,
                temperature=0.1,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            )
            answer = "".join(block.text for block in response_msg.content if hasattr(block, "text")) or answer
            anthropic_answered = True
        except Exception as exc:
            logger.warning(f"Recipe QA fallback (Anthropic error): {exc}")

    if not anthropic_answered or "unsure" in answer.lower():
        # Fallback: simple keyword search
        lowered = question.lower()
        instructions = recipe_payload["instructions"].split("\n")
        match = next((step for step in instructions if lowered in step.lower()), None)
        if match:
            answer = match
        else:
            answer = (
                "I could not find that detail in the recipe. "
                "Try asking about cook times, temperatures, or specific ingredients."
            )

    return {
        "recipe_id": recipe_id,
        "question": question,
        "answer": answer
    }


@app.post("/recipes/save", tags=["Recipes"])
async def save_recipe_endpoint(
    recipe_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save a recipe to user's collection"""
    recipe = db_save_recipe(db, current_user.id, recipe_data)
    logger.info(f"Recipe saved by {current_user.username}: {recipe_data.get('title')}")
    return {"message": "Recipe saved successfully", "recipe": recipe}


@app.post("/recipes/{recipe_id}/favorite", tags=["Recipes"])
async def toggle_favorite(
    recipe_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Toggle recipe favorite status"""
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id,
        Recipe.user_id == current_user.id
    ).first()
    
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    recipe.is_favorite = not recipe.is_favorite
    db.commit()
    
    return {"message": "Favorite status updated", "is_favorite": recipe.is_favorite}


# ==================== GROCERY LIST MANAGEMENT ====================

@app.get("/grocery-lists", tags=["Grocery"])
async def get_grocery_lists(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's grocery lists"""
    lists = db.query(GroceryList).filter(GroceryList.user_id == current_user.id).all()
    return {"lists": lists}


@app.post("/grocery-lists/{list_id}/complete", tags=["Grocery"])
async def complete_grocery_list(
    list_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark grocery list as completed"""
    from datetime import datetime
    
    grocery_list = db.query(GroceryList).filter(
        GroceryList.id == list_id,
        GroceryList.user_id == current_user.id
    ).first()
    
    if not grocery_list:
        raise HTTPException(status_code=404, detail="Grocery list not found")
    
    grocery_list.is_completed = True
    grocery_list.completed_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Grocery list marked as completed"}


# ==================== INVENTORY ENDPOINTS ====================

@app.get("/inventory", tags=["Inventory"])
async def get_inventory(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List inventory items"""
    items = get_inventory_items(db, current_user.id)
    return {"items": [_inventory_item_to_dict(item) for item in items]}


@app.post("/inventory", tags=["Inventory"], status_code=status.HTTP_201_CREATED)
async def create_inventory_item(
    payload: InventoryItemPayload,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new inventory entry"""
    expires_at, remaining = _compute_inventory_expiry(payload.expires_at, payload.days_until_expiry)
    item = upsert_inventory_item(
        db,
        current_user.id,
        {
            "name": payload.name,
            "quantity": payload.quantity or "1",
            "unit": payload.unit,
            "category": payload.category,
            "expires_at": expires_at,
            "days_until_expiry": remaining,
            "notes": payload.notes,
        }
    )

    if remaining <= 2:
        create_notification(
            db,
            current_user.id,
            title=f"{item.name} is almost out",
            message=f"{item.name} expires in {remaining} day(s).",
            notif_type="expiry",
            metadata={"inventory_item_id": item.id}
        )

    return {"item": _inventory_item_to_dict(item)}


@app.patch("/inventory/{item_id}", tags=["Inventory"])
async def update_inventory(
    item_id: int,
    payload: InventoryItemUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update an inventory item"""
    item = get_inventory_item(db, current_user.id, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Inventory item not found")

    update_data = payload.model_dump(exclude_unset=True)

    expiry_overrides = {}
    if "expires_at" in update_data or "days_until_expiry" in update_data:
        expires_at, remaining = _compute_inventory_expiry(
            update_data.get("expires_at"),
            update_data.get("days_until_expiry")
        )
        expiry_overrides["expires_at"] = expires_at
        expiry_overrides["days_until_expiry"] = remaining

    for field in ("name", "quantity", "unit", "category", "notes"):
        if field in update_data and update_data[field] is not None:
            setattr(item, field, update_data[field])

    for key, value in expiry_overrides.items():
        setattr(item, key, value)

    db.commit()
    db.refresh(item)

    if item.days_until_expiry is not None and item.days_until_expiry <= 2:
        create_notification(
            db,
            current_user.id,
            title=f"{item.name} is almost out",
            message=f"{item.name} expires in {item.days_until_expiry} day(s).",
            notif_type="expiry",
            metadata={"inventory_item_id": item.id}
        )

    return {"item": _inventory_item_to_dict(item)}


@app.delete("/inventory/{item_id}", tags=["Inventory"], status_code=status.HTTP_200_OK)
async def delete_inventory(
    item_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an inventory item"""
    success = delete_inventory_item(db, current_user.id, item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Inventory item not found")
    return {"message": "Inventory item removed"}


@app.post("/inventory/sync-from-grocery/{list_id}", tags=["Inventory"])
async def inventory_sync_from_grocery(
    list_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add unchecked grocery items to inventory"""
    grocery_list = db.query(GroceryList).filter(
        GroceryList.id == list_id,
        GroceryList.user_id == current_user.id
    ).first()

    if not grocery_list:
        raise HTTPException(status_code=404, detail="Grocery list not found")

    items = grocery_list.items or []
    added = 0
    updated = 0

    for entry in items:
        if entry.get("is_checked"):
            continue
        expires_at, remaining = _compute_inventory_expiry(None, 5)
        item = upsert_inventory_item(
            db,
            current_user.id,
            {
                "name": entry.get("name"),
                "quantity": entry.get("quantity") or "1",
                "unit": entry.get("unit"),
                "category": entry.get("category"),
                "expires_at": expires_at,
                "days_until_expiry": remaining,
                "notes": entry.get("description")
            }
        )
        if item:
            added += 1

    return {"added_count": added, "updated_count": updated}


# ==================== NOTIFICATIONS ENDPOINTS ====================

@app.get("/notifications", tags=["Notifications"])
async def list_notifications(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Return most recent notifications for the user"""
    notifications = get_notifications(db, current_user.id)
    return {"notifications": [_notification_to_dict(n) for n in notifications]}


@app.post("/notifications/{notification_id}/read", tags=["Notifications"])
async def mark_notification(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark notification as read"""
    notification = mark_notification_read(db, current_user.id, notification_id)
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"notification": _notification_to_dict(notification)}

# ==================== MEAL LOGGING ====================

@app.post("/meals/log", tags=["Meals"])
async def log_meal_endpoint(
    meal_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Log a meal in history"""
    meal = db_log_meal(db, current_user.id, meal_data)
    logger.info(f"Meal logged by {current_user.username}: {meal_data.get('recipe_title')}")
    return {"message": "Meal logged successfully", "meal": meal}


@app.get("/meals/history", tags=["Meals"])
async def get_meal_history(
    days: int = 7,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's meal history"""
    from datetime import datetime, timedelta
    
    since_date = datetime.utcnow() - timedelta(days=days)
    meals = db.query(MealHistory).filter(
        MealHistory.user_id == current_user.id,
        MealHistory.date >= since_date
    ).order_by(MealHistory.date.desc()).all()
    
    return {"meals": meals, "days": days}


# ==================== USER STATISTICS ====================

@app.get("/stats", tags=["User Stats"])
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user statistics"""
    from sqlalchemy import func
    
    total_recipes = db.query(func.count(Recipe.id)).filter(
        Recipe.user_id == current_user.id
    ).scalar()
    
    total_grocery_lists = db.query(func.count(GroceryList.id)).filter(
        GroceryList.user_id == current_user.id
    ).scalar()
    
    total_meals = db.query(func.count(MealHistory.id)).filter(
        MealHistory.user_id == current_user.id
    ).scalar()
    
    favorite_recipes = db.query(func.count(Recipe.id)).filter(
        Recipe.user_id == current_user.id,
        Recipe.is_favorite == True
    ).scalar()
    
    return {
        "total_recipes": total_recipes,
        "favorite_recipes": favorite_recipes,
        "total_grocery_lists": total_grocery_lists,
        "total_meals_logged": total_meals
    }


# ==================== AGENT METADATA ====================

@app.get("/agents-metadata", tags=["System"])
async def get_agents_metadata():
    """
    Returns agent metadata for Agentverse registration.
    Metadata is embedded in each agent's docstring following ASI:One best practices.
    
    Reference: https://docs.agentverse.ai/documentation/getting-started/overview
    """
    return {
        "agents": [
            {
                "name": "RecipeAgent",
                "handle": "@agentic-grocery-recipes",
                "description": "Intelligent recipe generator using Claude AI that creates personalized meal options based on user preferences, dietary goals, and macros",
                "tags": ["nutrition", "recipes", "meal-planning", "fetchai", "agentic-ai", "claude", "ai-powered"],
                "endpoint": "http://localhost:8000/recipe",
                "version": "0.3.0",
                "protocol": "chat-protocol-v0.3.0",
                "capabilities": ["recipe_generation", "macro_calculation", "dietary_personalization", "claude_integration"]
            },
            {
                "name": "GroceryAgent",
                "handle": "@agentic-grocery-shopping",
                "description": "Automated grocery list builder that extracts ingredients from recipes and uses Kroger API for real product pricing and availability",
                "tags": ["grocery", "shopping", "kroger", "fetchai", "agentic-ai", "automation", "e-commerce"],
                "endpoint": "http://localhost:8000/grocery",
                "version": "0.3.0",
                "protocol": "chat-protocol-v0.3.0",
                "capabilities": ["ingredient_extraction", "price_estimation", "kroger_integration", "list_generation"]
            }
        ],
        "system": {
            "framework": "Fetch.ai uAgents",
            "api_framework": "FastAPI",
            "llm": "Anthropic Claude",
            "grocery_api": "Kroger",
            "database": "SQLite + SQLAlchemy",
            "authentication": "JWT",
            "documentation": "https://github.com/yourusername/agentic-grocery"
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return {
        "error": "Not Found",
        "message": f"The endpoint {request.url.path} does not exist",
        "available_endpoints": {
            "public": ["/", "/health", "/agents-metadata", "/docs"],
            "auth": ["/auth/register", "/auth/login"],
            "user": ["/profile", "/stats"],
            "agents": ["/recipe", "/grocery"],
            "recipes": ["/recipes", "/recipes/save", "/recipes/{id}/favorite"],
            "grocery": ["/grocery-lists", "/grocery-lists/{id}/complete"],
            "meals": ["/meals/log", "/meals/history"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting Agentic Grocery API on {host}:{port}")
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    logger.info("🔍 Alternative docs: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

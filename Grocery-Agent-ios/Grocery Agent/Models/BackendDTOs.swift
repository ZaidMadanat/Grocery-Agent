import Foundation

// MARK: - Weekly Meals

struct WeeklyMealsRequest: Encodable {
    let start_date: String? // YYYY-MM-DD format
    let days: Int?
}

struct WeeklyMealsResponse: Decodable {
    let week: [WeekDay]
    let days: Int?
}

struct WeekDay: Decodable {
    let date: String
    let weekday: String
    let breakfast: MealPayload
    let lunch: MealPayload
    let dinner: MealPayload
    let snacks: MealPayload?
    let totals: DayTotals?
}

struct MealPayload: Decodable {
    let title: String
    let description: String?
    let cook_time: String?
    let servings: Int?
    let calories: Double?
    let protein: Double?
    let carbs: Double?
    let fat: Double?
    let instructions: String?
    let image_url: String?
    let ingredients: [IngredientPayload]?
}

struct IngredientPayload: Decodable {
    let name: String
    let quantity: String?
    let unit: String?
}

struct DayTotals: Decodable {
    let calories: Double
    let protein: Double
    let carbs: Double
    let fat: Double
}

// MARK: - User Profile

struct UserProfileResponse: Decodable {
    let id: Int
    let user_id: Int
    let height_cm: Double?
    let weight_kg: Double?
    let goal: String?
    let diet: String?
    let workout_frequency: String?
    let likes: [String]?
    let dislikes: [String]?
    let allergies: [String]?
    let target_protein_g: Double?
    let target_carbs_g: Double?
    let target_fat_g: Double?
    let target_calories: Double?
}

// MARK: - Recipes

struct RecipeGenerateRequest: Encodable {
    let preferences: RecipePreferences
    let user_profile: UserProfileDict?
    let context: String?
}

struct RecipePreferences: Encodable {
    let meal_type: String?
    let cook_time: String?
    let cuisine: String?
    let servings: Int?
}

struct UserProfileDict: Encodable {
    let height_cm: Double?
    let weight_kg: Double?
    let goal: String?
    let diet: String?
    let likes: [String]?
    let dislikes: [String]?
    let target_macros: TargetMacros?
}

struct TargetMacros: Encodable {
    let protein_g: Double?
    let carbs_g: Double?
    let fat_g: Double?
    let calories: Double?
}

struct RecipesResponse: Decodable {
    let recipes: [RecipeDetail]
    let agent: String?
}

struct RecipeDetailResponse: Decodable {
    let title: String
    let description: String?
    let cook_time: String?
    let servings: Int?
    let calories: Double?
    let protein_g: Double?
    let carbs_g: Double?
    let fat_g: Double?
    let instructions: String?
    let ingredients: [IngredientPayload]?
    let image_url: String?
}

// MARK: - Grocery Lists

struct GroceryListsResponse: Decodable {
    let lists: [GroceryListDetail]
}

struct GroceryListDetail: Decodable {
    let id: Int
    let name: String?
    let created_at: String
    let completed: Bool
    let items: [GroceryItemDetail]
}

struct GroceryItemDetail: Decodable {
    let id: Int
    let name: String
    let quantity: String?
    let price: Double?
    let product_id: String?
    let upc: String?
    let aisle: String?
    let is_checked: Bool
}

// MARK: - Inventory

struct InventoryResponse: Decodable {
    let items: [InventoryItemDetail]
}

struct InventoryItemDetail: Decodable {
    let id: Int
    let name: String
    let quantity: String
    let category: String?
    let expiry_date: String?
    let notes: String?
    let created_at: String
}

// MARK: - Notifications

struct NotificationsResponse: Decodable {
    let notifications: [NotificationDetail]
}

struct NotificationDetail: Decodable {
    let id: Int
    let type: String
    let title: String
    let message: String
    let created_at: String
    let is_read: Bool
}

// MARK: - Stats

struct StatsResponse: Decodable {
    let total_recipes_generated: Int?
    let total_grocery_lists: Int?
    let total_meals_logged: Int?
    let weekly_calories_avg: Double?
    let favorite_recipes: [String]?
}

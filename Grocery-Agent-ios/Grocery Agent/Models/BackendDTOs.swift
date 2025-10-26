import Foundation

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
    let totals: Totals?
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
}

struct Totals: Decodable {
    let calories: Double
    let protein: Double
    let carbs: Double
    let fat: Double
}


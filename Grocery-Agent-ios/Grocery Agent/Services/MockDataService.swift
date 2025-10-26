//
//  MockDataService.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Foundation

/// Provides deterministic mock data so the UI remains functional offline.
enum MockDataService {
    static func samplePreferences() -> UserPreferences {
        UserPreferences(
            dietaryRestrictions: [.vegetarian],
            customRestrictions: ["No mushrooms"],
            dailyCalorieGoal: 2100,
            mealTypes: [.breakfast, .lunch, .dinner, .snacks],
            macroPriorities: MacroPriorities(protein: 0.4, carbs: 0.35, fats: 0.25)
        )
    }

    static func weeklyPlan() -> [MealPlanDay] {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let mealTypes: [MealType] = [.breakfast, .lunch, .dinner, .snacks]

        return (0..<7).compactMap { offset in
            guard let date = calendar.date(byAdding: .day, value: offset, to: today) else { return nil }
            let meals = Dictionary(uniqueKeysWithValues: mealTypes.map { type -> (MealType, Meal) in
                let baseCalories = switch type {
                case .breakfast: 420
                case .lunch: 620
                case .dinner: 710
                case .snacks: 180
                }

                let macros = MacroBreakdown(
                    calories: baseCalories,
                    protein: Double(baseCalories) * 0.3 / 4,
                    carbs: Double(baseCalories) * 0.4 / 4,
                    fats: Double(baseCalories) * 0.3 / 9
                )

                let meal = Meal(
                    type: type,
                    name: sampleMealName(for: type, offset: offset),
                    description: "Chef-crafted \(type.rawValue.lowercased()) tuned to your macro goals.",
                    calories: baseCalories,
                    macros: macros,
                    imageName: sampleImage(for: type)
                )
                return (type, meal)
            })

            let totalCalories = meals.values.reduce(0) { $0 + $1.calories }
            let totalMacros = meals.values.reduce(MacroBreakdown.zero) { partial, meal in
                MacroBreakdown(
                    calories: partial.calories + meal.macros.calories,
                    protein: partial.protein + meal.macros.protein,
                    carbs: partial.carbs + meal.macros.carbs,
                    fats: partial.fats + meal.macros.fats
                )
            }

            return MealPlanDay(
                date: date,
                meals: meals,
                macros: MacroBreakdown(
                    calories: totalCalories,
                    protein: totalMacros.protein,
                    carbs: totalMacros.carbs,
                    fats: totalMacros.fats
                )
            )
        }
    }

    static func sampleRecipe(for meal: Meal) -> RecipeDetail {
        RecipeDetail(
            name: meal.name,
            summary: "A vibrant \(meal.type.rawValue.lowercased()) built from seasonal produce and whole grains.",
            heroImage: meal.imageName ?? "plate",
            calories: meal.calories,
            macros: meal.macros,
            ingredients: [
                .init(name: "Quinoa", quantity: "1 cup"),
                .init(name: "Roasted chickpeas", quantity: "1 cup"),
                .init(name: "Cherry tomatoes", quantity: "1/2 cup"),
                .init(name: "Baby spinach", quantity: "1 cup"),
                .init(name: "Tahini dressing", quantity: "2 tbsp")
            ],
            steps: [
                .init(title: "Prep Ingredients", instructions: "Rinse quinoa and prepare vegetables. Pat chickpeas dry.", durationMinutes: 5),
                .init(title: "Cook Quinoa", instructions: "Simmer quinoa in vegetable broth for 15 minutes.", durationMinutes: 15),
                .init(title: "Assemble Bowl", instructions: "Layer quinoa, vegetables, and chickpeas. Drizzle with tahini."),
                .init(title: "Serve", instructions: "Top with herbs and enjoy warm or chilled.")
            ]
        )
    }

    static func groceryItems() -> [GroceryItem] {
        [
            GroceryItem(name: "Baby Spinach", quantity: "1 bag", mealType: .lunch, dayLabel: "Monday"),
            GroceryItem(name: "Steel Cut Oats", quantity: "1 box", mealType: .breakfast, dayLabel: "Tuesday"),
            GroceryItem(name: "Salmon Filets", quantity: "4 pcs", mealType: .dinner, dayLabel: "Thursday"),
            GroceryItem(name: "Greek Yogurt", quantity: "2 tubs", mealType: .snacks, dayLabel: "Daily")
        ]
    }

    static func inventoryItems() -> [InventoryItem] {
        [
            InventoryItem(name: "Avocados", quantity: "3", daysUntilExpiry: 1, category: "Produce"),
            InventoryItem(name: "Almond Milk", quantity: "1 carton", daysUntilExpiry: 5, category: "Dairy"),
            InventoryItem(name: "Frozen Berries", quantity: "2 bags", daysUntilExpiry: 21, category: "Frozen"),
            InventoryItem(name: "Tofu", quantity: "2 blocks", daysUntilExpiry: 3, category: "Protein")
        ]
    }

    static func notifications() -> [AgentNotification] {
        [
            AgentNotification(
                title: "Avocados nearing expiry",
                message: "Use those avocados within the next 24 hours or freeze them for smoothies.",
                createdAt: Date().addingTimeInterval(-3600),
                type: .expiry
            ),
            AgentNotification(
                title: "Kroger cart ready",
                message: "Review your substitutions before checkout.",
                createdAt: Date().addingTimeInterval(-86000),
                type: .grocery
            ),
            AgentNotification(
                title: "New recipe suggestion",
                message: "Try the Roasted Cauliflower Tacos to use leftover produce.",
                createdAt: Date().addingTimeInterval(-172800),
                type: .recipe
            )
        ]
    }

    static func checkoutURL() -> URL {
        URL(string: "https://www.kroger.com/cl/cart")!
    }
}

private extension MockDataService {
    static func sampleMealName(for type: MealType, offset: Int) -> String {
        switch type {
        case .breakfast: return ["Sunrise Oat Parfait", "Protein Power Pancakes", "Spinach Feta Omelette"][offset % 3]
        case .lunch: return ["Mediterranean Grain Bowl", "Tahini Chickpea Wrap", "Miso Ginger Stir Fry"][offset % 3]
        case .dinner: return ["Herb Crusted Salmon", "Roasted Veggie Pasta", "Coconut Curry Lentils"][offset % 3]
        case .snacks: return ["Greek Yogurt Crunch", "Matcha Energy Bites", "Apple Almond Dippers"][offset % 3]
        }
    }

    static func sampleImage(for type: MealType) -> String {
        switch type {
        case .breakfast: return "breakfast"
        case .lunch: return "salad"
        case .dinner: return "dinner"
        case .snacks: return "snack"
        }
    }
}

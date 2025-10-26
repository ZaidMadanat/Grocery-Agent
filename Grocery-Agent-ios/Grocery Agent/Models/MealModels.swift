//
//  MealModels.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Foundation

/// Nutrition summary for a meal or daily plan.
struct MacroBreakdown: Codable, Hashable {
    var calories: Int
    var protein: Double
    var carbs: Double
    var fats: Double

    static let zero = MacroBreakdown(calories: 0, protein: 0, carbs: 0, fats: 0)

    var proteinPercentage: Double {
        guard calories > 0 else { return 0 }
        return protein * 4 / Double(calories)
    }

    var carbsPercentage: Double {
        guard calories > 0 else { return 0 }
        return carbs * 4 / Double(calories)
    }

    var fatsPercentage: Double {
        guard calories > 0 else { return 0 }
        return fats * 9 / Double(calories)
    }
}

/// Represents a single recipe returned by the agent.
struct Meal: Identifiable, Codable, Hashable {
    let id: UUID
    var type: MealType
    var name: String
    var description: String
    var calories: Int
    var macros: MacroBreakdown
    var imageName: String?
    var recipeId: UUID

    init(
        id: UUID = UUID(),
        type: MealType,
        name: String,
        description: String,
        calories: Int,
        macros: MacroBreakdown,
        imageName: String? = nil,
        recipeId: UUID = UUID()
    ) {
        self.id = id
        self.type = type
        self.name = name
        self.description = description
        self.calories = calories
        self.macros = macros
        self.imageName = imageName
        self.recipeId = recipeId
    }
}

/// Full set of meals for a given day.
struct MealPlanDay: Identifiable, Codable {
    let id: UUID
    var date: Date
    var meals: [MealType: Meal]
    var macros: MacroBreakdown

    init(
        id: UUID = UUID(),
        date: Date,
        meals: [MealType: Meal],
        macros: MacroBreakdown
    ) {
        self.id = id
        self.date = date
        self.meals = meals
        self.macros = macros
    }
}

/// Detailed recipe instruction for the cooking flow.
struct RecipeDetail: Identifiable, Codable {
    let id: UUID
    var name: String
    var summary: String
    var heroImage: String
    var calories: Int
    var macros: MacroBreakdown
    var ingredients: [Ingredient]
    var steps: [RecipeStep]

    init(
        id: UUID = UUID(),
        name: String,
        summary: String,
        heroImage: String,
        calories: Int,
        macros: MacroBreakdown,
        ingredients: [Ingredient],
        steps: [RecipeStep]
    ) {
        self.id = id
        self.name = name
        self.summary = summary
        self.heroImage = heroImage
        self.calories = calories
        self.macros = macros
        self.ingredients = ingredients
        self.steps = steps
    }

    struct Ingredient: Identifiable, Codable {
        let id: UUID
        var name: String
        var quantity: String

        init(id: UUID = UUID(), name: String, quantity: String) {
            self.id = id
            self.name = name
            self.quantity = quantity
        }
    }

    struct RecipeStep: Identifiable, Codable {
        let id: UUID
        var title: String
        var instructions: String
        var durationMinutes: Int?

        init(
            id: UUID = UUID(),
            title: String,
            instructions: String,
            durationMinutes: Int? = nil
        ) {
            self.id = id
            self.title = title
            self.instructions = instructions
            self.durationMinutes = durationMinutes
        }
    }
}

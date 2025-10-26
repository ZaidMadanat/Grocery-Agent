//
//  UserPreferences.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Foundation

/// Represents the core dietary selections captured during onboarding.
struct UserPreferences: Identifiable, Codable {
    let id: UUID
    var dietaryRestrictions: [DietaryRestriction]
    var customRestrictions: [String]
    var dailyCalorieGoal: Int
    var mealTypes: Set<MealType>
    var macroPriorities: MacroPriorities

    init(
        id: UUID = UUID(),
        dietaryRestrictions: [DietaryRestriction] = [],
        customRestrictions: [String] = [],
        dailyCalorieGoal: Int = 2000,
        mealTypes: Set<MealType> = Set(MealType.allCases),
        macroPriorities: MacroPriorities = .balanced
    ) {
        self.id = id
        self.dietaryRestrictions = dietaryRestrictions
        self.customRestrictions = customRestrictions
        self.dailyCalorieGoal = dailyCalorieGoal
        self.mealTypes = mealTypes
        self.macroPriorities = macroPriorities
    }
}

/// Predefined dietary options. Aligns with onboarding quick actions.
enum DietaryRestriction: String, CaseIterable, Codable, Identifiable {
    case vegetarian = "Vegetarian"
    case vegan = "Vegan"
    case glutenFree = "Gluten-Free"
    case other = "Other"

    var id: String { rawValue }

    var iconName: String {
        switch self {
        case .vegetarian: return "leaf"
        case .vegan: return "sparkle"
        case .glutenFree: return "allergens"
        case .other: return "square.and.pencil"
        }
    }
}

/// Supported meal buckets within a daily plan.
enum MealType: String, CaseIterable, Codable, Identifiable, Comparable, Hashable {
    case breakfast = "Breakfast"
    case lunch = "Lunch"
    case dinner = "Dinner"
    case snacks = "Snacks"

    var id: String { rawValue }

    static func < (lhs: MealType, rhs: MealType) -> Bool {
        Self.allCases.firstIndex(of: lhs) ?? 0 < Self.allCases.firstIndex(of: rhs) ?? 0
    }

    var symbolName: String {
        switch self {
        case .breakfast: return "sunrise.fill"
        case .lunch: return "fork.knife"
        case .dinner: return "moon.stars.fill"
        case .snacks: return "takeoutbag.and.cup.and.straw.fill"
        }
    }
}

/// Weighting for macronutrient emphasis within generated plans.
struct MacroPriorities: Codable, Equatable {
    var protein: Double
    var carbs: Double
    var fats: Double

    static let balanced = MacroPriorities(protein: 0.33, carbs: 0.34, fats: 0.33)

    var normalized: MacroPriorities {
        let total = max(protein + carbs + fats, 0.01)
        return MacroPriorities(
            protein: protein / total,
            carbs: carbs / total,
            fats: fats / total
        )
    }
}

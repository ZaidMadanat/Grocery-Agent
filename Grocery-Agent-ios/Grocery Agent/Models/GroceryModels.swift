//
//  GroceryModels.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Foundation

/// Consolidated grocery line item grouped by meal or day.
struct GroceryItem: Identifiable, Codable {
    let id: UUID
    var name: String
    var quantity: String
    var mealType: MealType
    var dayLabel: String
    var isChecked: Bool

    init(
        id: UUID = UUID(),
        name: String,
        quantity: String,
        mealType: MealType,
        dayLabel: String,
        isChecked: Bool = false
    ) {
        self.id = id
        self.name = name
        self.quantity = quantity
        self.mealType = mealType
        self.dayLabel = dayLabel
        self.isChecked = isChecked
    }
}

/// Inventory tracking for perishables and pantry items.
struct InventoryItem: Identifiable, Codable {
    let id: UUID
    var name: String
    var quantity: String
    var daysUntilExpiry: Int
    var category: String

    init(
        id: UUID = UUID(),
        name: String,
        quantity: String,
        daysUntilExpiry: Int,
        category: String
    ) {
        self.id = id
        self.name = name
        self.quantity = quantity
        self.daysUntilExpiry = daysUntilExpiry
        self.category = category
    }

    var statusColor: InventoryStatus {
        switch daysUntilExpiry {
        case ..<0: return .expired
        case 0...2: return .urgent
        case 3...5: return .warning
        default: return .fresh
        }
    }
}

enum InventoryStatus: String {
    case fresh
    case warning
    case urgent
    case expired
}

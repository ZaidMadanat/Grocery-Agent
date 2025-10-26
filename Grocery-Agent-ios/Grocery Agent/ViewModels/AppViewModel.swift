//
//  AppViewModel.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Combine
import Foundation

@MainActor
final class AppViewModel: ObservableObject {
    @Published private(set) var preferences: UserPreferences?
    @Published private(set) var weeklyPlan: [MealPlanDay] = []
    @Published private(set) var groceryItems: [GroceryItem] = []
    @Published private(set) var inventoryItems: [InventoryItem] = []
    @Published private(set) var notifications: [AgentNotification] = []

    private let preferenceStore: PreferenceStore

    init(preferenceStore: PreferenceStore) {
        self.preferenceStore = preferenceStore
        bootstrap()
    }

    init() {
        self.preferenceStore = PreferenceStore()
        bootstrap()
    }

    var isOnboarded: Bool {
        preferences != nil
    }

    func bootstrap() {
        if let stored = preferenceStore.loadPreferences() {
            preferences = stored
        } else {
            // Use sample values for new users until onboarding completes.
            preferences = nil
        }

        weeklyPlan = MockDataService.weeklyPlan()
        groceryItems = MockDataService.groceryItems()
        inventoryItems = MockDataService.inventoryItems()
        notifications = MockDataService.notifications()
    }

    func setPreferences(_ newPreferences: UserPreferences) {
        preferences = newPreferences
        preferenceStore.save(newPreferences)
    }

    func updatePreference(_ updateBlock: (inout UserPreferences) -> Void) {
        guard var existing = preferences else { return }
        updateBlock(&existing)
        setPreferences(existing)
    }

    func toggleGroceryItem(_ item: GroceryItem) {
        groceryItems = groceryItems.map { current in
            guard current.id == item.id else { return current }
            var modified = current
            modified.isChecked.toggle()
            return modified
        }
    }

    func updateGroceryItem(_ item: GroceryItem, quantity: String) {
        groceryItems = groceryItems.map { current in
            guard current.id == item.id else { return current }
            var modified = current
            modified.quantity = quantity
            return modified
        }
    }

    func updateInventoryItem(_ item: InventoryItem, amount: Int) {
        inventoryItems = inventoryItems.map { current in
            guard current.id == item.id else { return current }
            var updated = current
            updated.quantity = "\(amount)"
            return updated
        }
    }

    func updateMealPlan(_ plan: [MealPlanDay]) {
        weeklyPlan = plan
    }

    func removeGroceryItem(_ item: GroceryItem) {
        groceryItems.removeAll { $0.id == item.id }
    }

    func markNotificationRead(_ notification: AgentNotification) {
        notifications = notifications.map { current in
            guard current.id == notification.id else { return current }
            var updated = current
            updated.isRead = true
            return updated
        }
    }

    func resetOnboarding() {
        preferenceStore.reset()
        preferences = nil
    }
}

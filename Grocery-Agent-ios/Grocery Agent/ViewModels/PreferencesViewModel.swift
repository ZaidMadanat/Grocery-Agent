//
//  PreferencesViewModel.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Combine
import Foundation

@MainActor
final class PreferencesViewModel: ObservableObject {
    @Published var draft: UserPreferences
    @Published var isSaving = false

    private let appModel: AppViewModel

    init(appModel: AppViewModel) {
        self.appModel = appModel
        self.draft = appModel.preferences ?? MockDataService.samplePreferences()
    }

    func saveUpdates() {
        isSaving = true
        Task {
            try await Task.sleep(for: .milliseconds(450))
            appModel.setPreferences(draft)
            isSaving = false
        }
    }

    func toggleMeal(_ meal: MealType) {
        if draft.mealTypes.contains(meal) {
            draft.mealTypes.remove(meal)
        } else {
            draft.mealTypes.insert(meal)
        }
    }

    func toggleRestriction(_ restriction: DietaryRestriction) {
        if draft.dietaryRestrictions.contains(restriction) {
            draft.dietaryRestrictions.removeAll { $0 == restriction }
        } else {
            draft.dietaryRestrictions.append(restriction)
        }
    }

    func updateCustomRestrictions(_ value: [String]) {
        draft.customRestrictions = value
    }
}

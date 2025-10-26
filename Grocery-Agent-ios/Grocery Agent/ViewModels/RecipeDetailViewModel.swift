//
//  RecipeDetailViewModel.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Combine
import Foundation

@MainActor
final class RecipeDetailViewModel: ObservableObject {
    @Published private(set) var recipe: RecipeDetail
    @Published var currentStepIndex: Int = 0
    @Published var isCookingMode = false
    @Published var showVoiceHelp = false
    @Published var isMarkedCooked = false

    private let meal: Meal

    init(meal: Meal) {
        self.meal = meal
        self.recipe = MockDataService.sampleRecipe(for: meal)
    }

    var currentStep: RecipeDetail.RecipeStep? {
        guard recipe.steps.indices.contains(currentStepIndex) else { return nil }
        return recipe.steps[currentStepIndex]
    }

    func startCooking() {
        isCookingMode = true
        currentStepIndex = 0
    }

    func advanceStep() {
        guard currentStepIndex < recipe.steps.count - 1 else {
            isCookingMode = false
            return
        }
        currentStepIndex += 1
    }

    func rewindStep() {
        guard currentStepIndex > 0 else { return }
        currentStepIndex -= 1
    }

    func markAsCooked() {
        isMarkedCooked = true
    }
}

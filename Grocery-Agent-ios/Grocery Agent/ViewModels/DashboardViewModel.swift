//
//  DashboardViewModel.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Combine
import Foundation

@MainActor
final class DashboardViewModel: ObservableObject {
    @Published var selectedDayIndex: Int = 0
    @Published private(set) var week: [MealPlanDay] = []
    @Published private(set) var isRegenerating = false

    private let appModel: AppViewModel

    init(appModel: AppViewModel) {
        self.appModel = appModel
        week = appModel.weeklyPlan
    }

    var selectedDay: MealPlanDay? {
        guard week.indices.contains(selectedDayIndex) else { return nil }
        return week[selectedDayIndex]
    }

    var dayLabels: [String] {
        week.map { DateFormatter.weekdayFormatter.string(from: $0.date) }
    }

    func refreshPlan() {
        guard !isRegenerating else { return }
        isRegenerating = true
        Task {
            try await Task.sleep(for: .seconds(1))
            let refreshed = MockDataService.weeklyPlan().shuffled()
            appModel.updateMealPlan(refreshed)
            week = refreshed
            isRegenerating = false
        }
    }

    func meal(for type: MealType) -> Meal? {
        selectedDay?.meals[type]
    }

    func macroGoal(for macro: String) -> Double {
        guard let preferences = appModel.preferences else { return 0 }
        switch macro {
        case "protein":
            return Double(preferences.dailyCalorieGoal) * preferences.macroPriorities.protein / 4
        case "carbs":
            return Double(preferences.dailyCalorieGoal) * preferences.macroPriorities.carbs / 4
        case "fats":
            return Double(preferences.dailyCalorieGoal) * preferences.macroPriorities.fats / 9
        default:
            return 0
        }
    }
}

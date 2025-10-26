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
            do {
                let client = APIClient(tokenProvider: { AuthService.shared.getToken() })
                struct Payload: Encodable { let days: Int }
                let resp: WeeklyMealsResponse = try await client.postJSON(
                    "/weekly-meals/generate",
                    payload: Payload(days: 7)
                )

                let iso = ISO8601DateFormatter()
                func toMeal(_ p: MealPayload, type: MealType) -> Meal {
                    let macros = MacroBreakdown(
                        calories: Int(p.calories ?? 0),
                        protein: p.protein ?? 0,
                        carbs: p.carbs ?? 0,
                        fats: p.fat ?? 0
                    )
                    return Meal(
                        type: type,
                        name: p.title,
                        description: p.description ?? "",
                        calories: Int(p.calories ?? 0),
                        macros: macros,
                        imageName: nil
                    )
                }

                let mapped: [MealPlanDay] = resp.week.map { day in
                    let date = iso.date(from: day.date) ?? Date()
                    let meals: [MealType: Meal] = [
                        .breakfast: toMeal(day.breakfast, type: .breakfast),
                        .lunch: toMeal(day.lunch, type: .lunch),
                        .dinner: toMeal(day.dinner, type: .dinner)
                    ]
                    let totals = day.totals
                    let dm = MacroBreakdown(
                        calories: Int(totals?.calories ?? 0),
                        protein: totals?.protein ?? 0,
                        carbs: totals?.carbs ?? 0,
                        fats: totals?.fat ?? 0
                    )
                    return MealPlanDay(date: date, meals: meals, macros: dm)
                }

                await MainActor.run {
                    self.appModel.updateMealPlan(mapped)
                    self.week = mapped
                    self.isRegenerating = false
                }
            } catch {
                await MainActor.run { self.isRegenerating = false }
                print("[Dashboard] weekly-meals error:", error)
            }
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

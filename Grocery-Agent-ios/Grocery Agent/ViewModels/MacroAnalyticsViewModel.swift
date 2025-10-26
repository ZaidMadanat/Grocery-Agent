//
//  MacroAnalyticsViewModel.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Combine
import Foundation

@MainActor
final class MacroAnalyticsViewModel: ObservableObject {
    @Published private(set) var weeklyPlan: [MealPlanDay]

    private let appModel: AppViewModel
    private var cancellables = Set<AnyCancellable>()

    init(appModel: AppViewModel) {
        self.appModel = appModel
        self.weeklyPlan = appModel.weeklyPlan

        appModel.$weeklyPlan
            .receive(on: DispatchQueue.main)
            .sink { [weak self] plan in
                self?.weeklyPlan = plan
            }
            .store(in: &cancellables)
    }

    struct MacroDataPoint: Identifiable {
        let id: String
        let day: String
        let macro: MacroType
        let value: Double
    }

    enum MacroType: String, CaseIterable {
        case calories = "Calories"
        case protein = "Protein"
        case carbs = "Carbs"
        case fats = "Fats"

        var unit: String {
            switch self {
            case .calories: return "kcals"
            case .protein, .carbs, .fats: return "g"
            }
        }
    }

    func data(for macro: MacroType) -> [MacroDataPoint] {
        let calendar = Calendar.current

        let grouped = Dictionary(grouping: weeklyPlan) { plan in
            calendar.startOfDay(for: plan.date)
        }

        return grouped
            .keys
            .sorted()
            .compactMap { day -> MacroDataPoint? in
                guard let entries = grouped[day] else { return nil }

                let totals = entries.reduce((calories: 0.0, protein: 0.0, carbs: 0.0, fats: 0.0)) { partial, plan in
                    (
                        calories: partial.calories + Double(plan.macros.calories),
                        protein: partial.protein + plan.macros.protein,
                        carbs: partial.carbs + plan.macros.carbs,
                        fats: partial.fats + plan.macros.fats
                    )
                }

                let label = DateFormatter.weekdayFormatter.string(from: day)
                let value: Double = {
                    switch macro {
                    case .calories: return totals.calories
                    case .protein: return totals.protein
                    case .carbs: return totals.carbs
                    case .fats: return totals.fats
                    }
                }()

                return MacroDataPoint(id: "\(macro.rawValue)-\(label)", day: label, macro: macro, value: value)
            }
    }

    func trend(for macro: MacroType) -> Double {
        let values = data(for: macro).map(\.value)
        guard let first = values.first, let last = values.last else { return 0 }
        return last - first
    }
}

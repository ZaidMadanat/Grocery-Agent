//
//  PreferencesView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct PreferencesView: View {
    @StateObject private var viewModel: PreferencesViewModel
    @FocusState private var customInputFocused: Bool

    init(appModel: AppViewModel) {
        _viewModel = StateObject(wrappedValue: PreferencesViewModel(appModel: appModel))
    }

    var body: some View {
        NavigationStack {
            Form {
                dietarySection
                goalsSection
                macrosSection
            }
            .navigationTitle("Preferences")
#if os(iOS)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(action: viewModel.saveUpdates) {
                        if viewModel.isSaving {
                            ProgressView()
                        } else {
                            Text("Save")
                                .bold()
                        }
                    }
                    .disabled(viewModel.isSaving)
                }
            }
#else
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    Button(action: viewModel.saveUpdates) {
                        if viewModel.isSaving {
                            ProgressView()
                        } else {
                            Text("Save")
                                .bold()
                        }
                    }
                    .disabled(viewModel.isSaving)
                }
            }
#endif
            .formStyle(.grouped)
        }
    }
}

private extension PreferencesView {
    var dietarySection: some View {
        Section("Dietary Restrictions") {
            let columns = [GridItem(.adaptive(minimum: 140), spacing: 8)]
            LazyVGrid(columns: columns, spacing: 8) {
                ForEach(DietaryRestriction.allCases) { restriction in
                    SelectableChip(
                        title: restriction.rawValue,
                        systemImage: restriction.iconName,
                        isSelected: viewModel.draft.dietaryRestrictions.contains(restriction)
                    ) {
                        viewModel.toggleRestriction(restriction)
                    }
                }
            }
            .padding(.vertical, 4)

            VStack(alignment: .leading, spacing: 8) {
                Text("Custom".uppercased())
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.secondary)

                TextField(
                    "Comma separated custom restrictions",
                    text: Binding(
                        get: { viewModel.draft.customRestrictions.joined(separator: ", ") },
                        set: { newValue in
                            let parsed = newValue
                                .split(separator: ",")
                                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                                .filter { !$0.isEmpty }
                            viewModel.updateCustomRestrictions(parsed)
                        }
                    ),
                    axis: .vertical
                )
                .textFieldStyle(.roundedBorder)
                .focused($customInputFocused)
            }
            .padding(.vertical, 4)
        }
    }

    var goalsSection: some View {
        Section("Daily Goals") {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label("Calorie Target", systemImage: "flame.fill")
                    Spacer()
                    Text("\(viewModel.draft.dailyCalorieGoal) kcal")
                        .font(.headline)
                }
                Slider(
                    value: Binding(
                        get: { Double(viewModel.draft.dailyCalorieGoal) },
                        set: { viewModel.draft.dailyCalorieGoal = Int($0) }
                    ),
                    in: 1200...3500,
                    step: 50
                )
                .tint(.orange)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Meals In Plan")
                    .font(.subheadline.bold())
                let columns = [GridItem(.adaptive(minimum: 140), spacing: 8)]
                LazyVGrid(columns: columns, spacing: 8) {
                    ForEach(MealType.allCases) { meal in
                        SelectableChip(
                            title: meal.rawValue,
                            systemImage: meal.symbolName,
                            isSelected: viewModel.draft.mealTypes.contains(meal)
                        ) {
                            viewModel.toggleMeal(meal)
                        }
                    }
                }
                .padding(.vertical, 4)
            }
        }
    }

    var macrosSection: some View {
        Section("Macro Priorities") {
            MacroSlidersView(
                protein: viewModel.draft.macroPriorities.protein,
                carbs: viewModel.draft.macroPriorities.carbs,
                fats: viewModel.draft.macroPriorities.fats
            ) { protein, carbs, fats in
                viewModel.draft.macroPriorities = MacroPriorities(
                    protein: protein,
                    carbs: carbs,
                    fats: fats
                )
            }
            .padding(.vertical, 8)
        }
    }
}

#Preview {
    PreferencesView(appModel: AppViewModel())
}

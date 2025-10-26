//
//  DashboardView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct DashboardView: View {
    @ObservedObject var appModel: AppViewModel
    @StateObject private var viewModel: DashboardViewModel
    @State private var selectedMeal: Meal?
    @State private var showingPreferences = false

    init(appModel: AppViewModel) {
        _viewModel = StateObject(wrappedValue: DashboardViewModel(appModel: appModel))
        self.appModel = appModel
    }

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottomTrailing) {
                ScrollView {
                    VStack(spacing: 24) {
                        header
                        dayPicker
                        macroSection
                        mealSection
                    }
                    .padding(.horizontal, 24)
                    .padding(.bottom, 120)
                }

                FloatingActionButton(
                    icon: viewModel.isRegenerating ? "hourglass" : "sparkles",
                    title: viewModel.isRegenerating ? "Regenerating..." : "Regenerate"
                ) {
                    viewModel.refreshPlan()
                }
                .opacity(viewModel.isRegenerating ? 0.6 : 1)
                .disabled(viewModel.isRegenerating)
                .padding(.trailing, 24)
                .padding(.bottom, 32)
            }
            .background(Color.agentGroupedBackground.ignoresSafeArea())
            .navigationDestination(item: $selectedMeal) { meal in
                RecipeDetailView(meal: meal, appModel: appModel)
            }
#if os(iOS)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showingPreferences.toggle()
                    } label: {
                        Label("Preferences", systemImage: "slider.horizontal.3")
                    }
                    .labelStyle(.iconOnly)
                }
            }
#else
            .toolbar {
                ToolbarItem(placement: .automatic) {
                    Button {
                        showingPreferences.toggle()
                    } label: {
                        Label("Preferences", systemImage: "slider.horizontal.3")
                    }
                    .labelStyle(.iconOnly)
                }
            }
#endif
            .sheet(isPresented: $showingPreferences) {
                PreferencesView(appModel: appModel)
            }
            .navigationTitle("Dashboard")
        }
    }
}

private extension DashboardView {
    var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("This week")
                .font(.largeTitle.bold())
            if let preferences = appModel.preferences {
                Text("Tracking \(Int(preferences.dailyCalorieGoal)) kcals · \(preferences.mealTypes.count) meals per day")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            } else {
                Text("We'll personalize once you finish setup.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    var dayPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(Array(viewModel.dayLabels.enumerated()), id: \.offset) { index, label in
                    let day = viewModel.week.indices.contains(index) ? viewModel.week[index] : nil
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                            viewModel.selectedDayIndex = index
                        }
                    } label: {
                        VStack(spacing: 6) {
                            Text(label)
                                .font(.headline)
                            if let date = day?.date {
                                Text(date.formatted(date: .abbreviated, time: .omitted))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .background(dayBackground(isSelected: viewModel.selectedDayIndex == index))
                        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 4)
        }
    }

    func dayBackground(isSelected: Bool) -> some ShapeStyle {
        isSelected ? AnyShapeStyle(LinearGradient(colors: [.blue.opacity(0.8), .green.opacity(0.75)], startPoint: .leading, endPoint: .trailing)) :
            AnyShapeStyle(Color.agentSecondaryBackground)
    }

    var macroSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Daily macros")
                    .font(.title2.bold())
                Spacer()
                if let macros = viewModel.selectedDay?.macros {
                    Text("\(macros.calories) kcals logged")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 24) {
                    if let day = viewModel.selectedDay {
                        MacroRing(
                            title: "Calories",
                            value: Double(day.macros.calories),
                            goal: Double(appModel.preferences?.dailyCalorieGoal ?? 2000),
                            gradient: Gradient(colors: [.purple, .blue]),
                            unit: "kcals"
                        )
                        MacroRing(
                            title: "Protein",
                            value: day.macros.protein,
                            goal: viewModel.macroGoal(for: "protein"),
                            gradient: Gradient(colors: [.orange, .pink]),
                            unit: "g"
                        )
                        MacroRing(
                            title: "Carbs",
                            value: day.macros.carbs,
                            goal: viewModel.macroGoal(for: "carbs"),
                            gradient: Gradient(colors: [.mint, .green]),
                            unit: "g"
                        )
                        MacroRing(
                            title: "Fats",
                            value: day.macros.fats,
                            goal: viewModel.macroGoal(for: "fats"),
                            gradient: Gradient(colors: [.yellow, .orange]),
                            unit: "g"
                        )
                    }
                }
                .padding(.vertical, 8)
            }
        }
    }

    var mealSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Meals")
                .font(.title2.bold())

            if let selected = viewModel.selectedDay {
                ForEach(MealType.allCases) { type in
                    if let meal = selected.meals[type] {
                        MealCardView(meal: meal) {
                            selectedMeal = meal
                        }
                    }
                }
            } else {
                ContentUnavailableView("No plan yet", systemImage: "calendar.badge.exclamationmark", description: Text("Generate first to view your daily meals."))
            }
        }
    }
}

#Preview {
    DashboardView(appModel: AppViewModel())
}

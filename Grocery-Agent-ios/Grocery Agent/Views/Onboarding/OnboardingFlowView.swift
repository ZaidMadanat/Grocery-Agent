//
//  OnboardingFlowView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct OnboardingFlowView: View {
    @ObservedObject var appModel: AppViewModel
    @StateObject private var viewModel = OnboardingViewModel()
    @FocusState private var customRestrictionFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                header
                progressIndicator

                Group {
                    switch viewModel.step {
                    case .welcome:
                        welcome
                    case .dietary:
                        dietary
                    case .caloriesAndMeals:
                        caloriesAndMeals
                    case .macros:
                        macros
                    case .review:
                        review
                    }
                }
                .animation(.easeInOut, value: viewModel.step)

                if viewModel.showValidationError {
                    validationBanner
                        .transition(.opacity.combined(with: .slide))
                }

                footerButtons
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 32)
            .background(
                LinearGradient(
                    colors: [.mint.opacity(0.1), .blue.opacity(0.05)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
            )
            .toolbar(.hidden, for: .navigationBar)
        }
    }
}

private extension OnboardingFlowView {
    var header: some View {
        VStack(spacing: 8) {
            Text(viewModel.step.title)
                .font(.largeTitle.bold())
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(viewModel.step.subtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    var progressIndicator: some View {
        VStack(alignment: .leading, spacing: 12) {
            ProgressView(value: Double(viewModel.step.rawValue), total: Double(OnboardingViewModel.Step.allCases.count - 1))
                .tint(.accentColor)
            HStack {
                Text("#\(viewModel.step.rawValue + 1) of \(OnboardingViewModel.Step.allCases.count)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
            }
        }
    }

    var footerButtons: some View {
        HStack(spacing: 16) {
            if viewModel.step != .welcome {
                Button("Back", action: viewModel.goBack)
                    .buttonStyle(.borderedProminent)
                    .tint(.secondary.opacity(0.3))
            }

            Button(action: continueAction) {
                Text(viewModel.step == .review ? "Finish Setup" : "Continue")
                    .bold()
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
        }
    }

    var validationBanner: some View {
        Label("Complete the required fields to continue.", systemImage: "exclamationmark.triangle.fill")
            .lineLimit(2)
            .padding()
            .frame(maxWidth: .infinity)
            .background(.orange.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    func continueAction() {
        if viewModel.step == .review {
            let preferences = viewModel.finalizePreferences()
            appModel.setPreferences(preferences)
        } else {
            viewModel.advance()
        }
    }

    @ViewBuilder
    var welcome: some View {
        VStack(alignment: .leading, spacing: 24) {
            Image(systemName: "cart.circle.fill")
                .font(.system(size: 72))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(.green)
            Text("GroceryAgent wires Letta, Groq, and Kroger together to plan, shop, and cook for you.")
                .font(.title3)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 12) {
                Label("Curated weekly meal plans tuned to your macros.", systemImage: "fork.knife.circle.fill")
                Label("One-tap Kroger checkout with smart substitutions.", systemImage: "cart.fill.badge.plus")
                Label("Inventory intelligence so nothing spoils unnoticed.", systemImage: "leaf.fill")
            }
            .labelStyle(.onboarding)
            Spacer()
        }
        .tag(OnboardingViewModel.Step.welcome)
    }

    @ViewBuilder
    var dietary: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Select the dietary patterns that best match your needs.")
                .font(.headline)
            let columns = [GridItem(.adaptive(minimum: 140), spacing: 12)]
            LazyVGrid(columns: columns, spacing: 12) {
                ForEach(DietaryRestriction.allCases) { restriction in
                    SelectableChip(
                        title: restriction.rawValue,
                        systemImage: restriction.iconName,
                        isSelected: viewModel.selectedRestrictions.contains(restriction)
                    ) {
                        withAnimation {
                            viewModel.toggleRestriction(restriction)
                        }
                    }
                }
            }

            if viewModel.selectedRestrictions.contains(.other) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Tell us more")
                        .font(.subheadline.bold())
                    TextField("e.g. No mushrooms, low sodium", text: $viewModel.otherRestrictionInput, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .focused($customRestrictionFocused)
                        .onChange(of: viewModel.otherRestrictionInput) { _ in
                            viewModel.customRestrictions = viewModel.parsedCustomRestrictions
                        }
                    Text("Separate restrictions with commas.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
        .tag(OnboardingViewModel.Step.dietary)
    }

    @ViewBuilder
    var caloriesAndMeals: some View {
        VStack(alignment: .leading, spacing: 24) {
            VStack(alignment: .leading) {
                Text("Daily Calorie Goal")
                    .font(.headline)
                HStack(alignment: .firstTextBaseline) {
                    Text("\(Int(viewModel.dailyCalorieGoal))")
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                    Text("kcals")
                        .font(.headline)
                        .foregroundStyle(.secondary)
                }
                Slider(value: $viewModel.dailyCalorieGoal, in: 1200...3500, step: 50)
                    .tint(.mint)
                Text("We'll pull balanced recipes that average to this target.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 12) {
                Text("Meals to Include")
                    .font(.headline)
                let columns = [GridItem(.adaptive(minimum: 140), spacing: 12)]
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(MealType.allCases) { mealType in
                        SelectableChip(
                            title: mealType.rawValue,
                            systemImage: mealType.symbolName,
                            isSelected: viewModel.mealTypes.contains(mealType)
                        ) {
                            withAnimation {
                                viewModel.toggleMealType(mealType)
                            }
                        }
                    }
                }
            }
            Spacer()
        }
        .tag(OnboardingViewModel.Step.caloriesAndMeals)
    }

    @ViewBuilder
    var macros: some View {
        VStack(alignment: .leading, spacing: 24) {
            Text("Highlight specific macros and we'll prioritize recipes that support them.")
                .font(.headline)

            MacroSlidersView(
                protein: viewModel.macroFocus.protein,
                carbs: viewModel.macroFocus.carbs,
                fats: viewModel.macroFocus.fats
            ) { protein, carbs, fats in
                viewModel.updateMacroFocus(protein: protein, carbs: carbs, fats: fats)
            }

            Spacer()
        }
        .tag(OnboardingViewModel.Step.macros)
    }

    @ViewBuilder
    var review: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                summaryCard(
                    title: "Dietary Profile",
                    icon: "leaf.fill"
                ) {
                    dietarySummary
                }

                summaryCard(title: "Daily Goal", icon: "flame.fill") {
                    Text("\(Int(viewModel.dailyCalorieGoal)) calories · \(viewModel.mealTypes.count) meals")
                    Text(viewModel.mealTypes.map(\.rawValue).sorted().joined(separator: " • "))
                        .foregroundStyle(.secondary)
                }

                summaryCard(title: "Macro Focus", icon: "chart.pie.fill") {
                    let normalized = viewModel.macroFocus.normalized
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Protein \(Int(normalized.protein * 100))%")
                        Text("Carbs \(Int(normalized.carbs * 100))%")
                        Text("Fats \(Int(normalized.fats * 100))%")
                    }
                }
            }
            .padding(.top, 12)
        }
        .tag(OnboardingViewModel.Step.review)
    }

    func summaryCard(title: String, icon: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: icon)
                .font(.headline)
            content()
                .font(.subheadline)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.agentSecondaryBackground.opacity(0.65), in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .stroke(Color.white.opacity(0.15))
        )
    }

    @ViewBuilder
    var dietarySummary: some View {
        VStack(alignment: .leading, spacing: 4) {
            if !viewModel.selectedRestrictions.filter({ $0 != .other }).isEmpty {
                Text(viewModel.selectedRestrictions.filter { $0 != .other }.map(\.rawValue).joined(separator: ", "))
            }

            if !viewModel.parsedCustomRestrictions.isEmpty {
                Text("Custom: \(viewModel.parsedCustomRestrictions.joined(separator: ", "))")
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private extension LabelStyle where Self == OnboardingLabelStyle {
    static var onboarding: OnboardingLabelStyle { .init() }
}

private struct OnboardingLabelStyle: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 12) {
            configuration.icon
                .foregroundStyle(Color.accentColor)
            configuration.title
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

#Preview {
    OnboardingFlowView(appModel: AppViewModel())
}

//
//  RecipeDetailView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct RecipeDetailView: View {
    let meal: Meal
    @ObservedObject var appModel: AppViewModel
    @StateObject private var viewModel: RecipeDetailViewModel

    init(meal: Meal, appModel: AppViewModel) {
        self.meal = meal
        self.appModel = appModel
        _viewModel = StateObject(wrappedValue: RecipeDetailViewModel(meal: meal))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                heroCard
                macroHighlights
                ingredientsSection
                if viewModel.isCookingMode {
                    cookingStepper
                } else {
                    stepsPreview
                }
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 40)
        }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    viewModel.startCooking()
                } label: {
                    Label("Start Cooking", systemImage: "play.circle.fill")
                }
            }
        }
        .toolbar {
            ToolbarItem(placement: .bottomBar) {
                Button {
                    viewModel.markAsCooked()
                } label: {
                    Label(viewModel.isMarkedCooked ? "Cooked!" : "Mark as cooked", systemImage: "checkmark.circle")
                        .font(.headline)
                }
                .disabled(viewModel.isMarkedCooked)
            }
        }
        .navigationTitle(meal.name)
        .navigationBarTitleDisplayMode(.inline)
        .background(Color.agentGroupedBackground.ignoresSafeArea())
    }
}

private extension RecipeDetailView {
    var heroCard: some View {
        ZStack(alignment: .bottomLeading) {
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [.purple, .blue.opacity(0.8)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(height: 220)
            VStack(alignment: .leading, spacing: 12) {
                Text(meal.type.rawValue.uppercased())
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
                Text(meal.name)
                    .font(.title.bold())
                    .foregroundStyle(.white)
                Text(viewModel.recipe.summary)
                    .foregroundStyle(.white.opacity(0.85))
                Label("Live voice help", systemImage: viewModel.showVoiceHelp ? "waveform.circle.fill" : "waveform")
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(.thinMaterial, in: Capsule())
                    .onTapGesture {
                        withAnimation(.spring) {
                            viewModel.showVoiceHelp.toggle()
                        }
                    }
            }
            .padding(24)
        }
    }

    var macroHighlights: some View {
        HStack(spacing: 20) {
            StatBlock(title: "Calories", value: "\(meal.macros.calories)", unit: "kcals", color: .purple)
            StatBlock(title: "Protein", value: "\(Int(meal.macros.protein))", unit: "g", color: .orange)
            StatBlock(title: "Carbs", value: "\(Int(meal.macros.carbs))", unit: "g", color: .mint)
            StatBlock(title: "Fats", value: "\(Int(meal.macros.fats))", unit: "g", color: .pink)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .scrollIndicators(.hidden)
    }

    var ingredientsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Ingredients")
                .font(.title2.bold())
            ForEach(viewModel.recipe.ingredients) { ingredient in
                HStack {
                    Image(systemName: "checkmark.seal")
                        .foregroundStyle(.green)
                    Text(ingredient.name)
                    Spacer()
                    Text(ingredient.quantity)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }
        }
    }

    var stepsPreview: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Steps")
                .font(.title2.bold())
            ForEach(viewModel.recipe.steps) { step in
                VStack(alignment: .leading, spacing: 8) {
                    Text(step.title)
                        .font(.headline)
                    Text(step.instructions)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding()
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
        }
    }

    var cookingStepper: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Cooking mode")
                .font(.title2.bold())

            if let currentStep = viewModel.currentStep {
                VStack(alignment: .leading, spacing: 12) {
                    Text(currentStep.title)
                        .font(.headline)
                    Text(currentStep.instructions)
                        .font(.body)
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))

                HStack {
                    Button {
                        viewModel.rewindStep()
                    } label: {
                        Label("Back", systemImage: "arrow.left")
                    }
                    .buttonStyle(.bordered)

                    Spacer()

                    Button {
                        viewModel.advanceStep()
                    } label: {
                        Label("Next", systemImage: "arrow.right")
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding()
        .background(Color.agentSecondaryBackground, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
    }
}

private struct StatBlock: View {
    let title: String
    let value: String
    let unit: String
    let color: Color

    var body: some View {
        VStack(spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline)
                .foregroundStyle(color)
            Text(unit)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.agentSecondaryBackground, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

#Preview {
    RecipeDetailView(meal: MockDataService.weeklyPlan().first!.meals[.lunch]!, appModel: AppViewModel())
}

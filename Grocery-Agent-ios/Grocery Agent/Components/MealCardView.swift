//
//  MealCardView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct MealCardView: View {
    let meal: Meal
    var action: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(meal.type.rawValue.uppercased())
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundStyle(.secondary)
                    Text(meal.name)
                        .font(.headline)
                        .foregroundStyle(.primary)
                    Text(meal.description)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer()
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(Color.accentColor.opacity(0.12))
                    .frame(width: 72, height: 72)
                    .overlay {
                        Image(systemName: meal.type.symbolName)
                            .font(.title2)
                            .foregroundStyle(Color.accentColor)
                    }
            }

            HStack(spacing: 16) {
                macroPill(title: "Calories", value: "\(meal.calories)")
                macroPill(title: "Protein", value: "\(Int(meal.macros.protein))g")
                macroPill(title: "Carbs", value: "\(Int(meal.macros.carbs))g")
            }

            Button(action: action) {
                Label("View details", systemImage: "chevron.right.circle.fill")
                    .font(.subheadline.weight(.semibold))
            }
            .buttonStyle(.borderless)
            .tint(.accentColor)
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .shadow(color: Color.black.opacity(0.06), radius: 12, x: 0, y: 8)
    }

    private func macroPill(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.bold())
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.agentSecondaryBackground, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

#Preview {
    MealCardView(meal: MockDataService.weeklyPlan().first!.meals[.lunch]!) {}
        .padding()
        .background(Color.agentGroupedBackground)
}

//
//  MacroSlidersView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct MacroSlidersView: View {
    @State private var protein: Double
    @State private var carbs: Double
    @State private var fats: Double

    let onChange: (Double, Double, Double) -> Void

    init(
        protein: Double,
        carbs: Double,
        fats: Double,
        onChange: @escaping (Double, Double, Double) -> Void
    ) {
        _protein = State(initialValue: protein)
        _carbs = State(initialValue: carbs)
        _fats = State(initialValue: fats)
        self.onChange = onChange
    }

    var body: some View {
        VStack(spacing: 24) {
            metricRow(
                title: "Protein",
                icon: "bolt.fill",
                color: .orange,
                value: protein
            ) {
                protein = $0
                propagate()
            }

            metricRow(
                title: "Carbs",
                icon: "leaf.fill",
                color: .mint,
                value: carbs
            ) {
                carbs = $0
                propagate()
            }

            metricRow(
                title: "Fats",
                icon: "drop.fill",
                color: .pink,
                value: fats
            ) {
                fats = $0
                propagate()
            }

            totalSummary
        }
        .onAppear(perform: propagate)
    }

    private var totalSummary: some View {
        let total = max(protein + carbs + fats, 0.01)
        let normalizedProtein = protein / total
        let normalizedCarbs = carbs / total
        let normalizedFats = fats / total

        return VStack(alignment: .leading, spacing: 16) {
            Text("Macro Balance")
                .font(.title3.bold())

            VStack(spacing: 12) {
                macroSummaryRow(title: "Protein", symbol: "bolt.fill", value: normalizedProtein, color: .orange)
                macroSummaryRow(title: "Carbs", symbol: "leaf.fill", value: normalizedCarbs, color: .mint)
                macroSummaryRow(title: "Fats", symbol: "drop.fill", value: normalizedFats, color: .pink)
            }

            Text("Total weight \(Int(total * 100))")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func metricRow(
        title: String,
        icon: String,
        color: Color,
        value: Double,
        onChange: @escaping (Double) -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: icon)
                .font(.headline)
                .foregroundStyle(color)
            Slider(value: Binding(
                get: { value },
                set: { newValue in
                    onChange(newValue)
                }
            ), in: 0...1, step: 0.05)
            .tint(color)
        }
    }

    private func macroSummaryRow(title: String, symbol: String, value: Double, color: Color) -> some View {
        HStack(spacing: 16) {
            Circle()
                .fill(color.opacity(0.2))
                .frame(width: 34, height: 34)
                .overlay {
                    Image(systemName: symbol)
                        .font(.footnote)
                        .foregroundStyle(color)
                }

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(title)
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    Text("\(Int(value * 100))%")
                        .font(.subheadline.weight(.semibold))
                }

                ProgressView(value: value)
                    .tint(color)
                    .frame(maxWidth: .infinity)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color.agentSecondaryBackground.opacity(0.4), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private func propagate() {
        onChange(protein, carbs, fats)
    }
}

#Preview {
    MacroSlidersView(protein: 0.4, carbs: 0.35, fats: 0.25, onChange: { _, _, _ in })
        .padding()
}

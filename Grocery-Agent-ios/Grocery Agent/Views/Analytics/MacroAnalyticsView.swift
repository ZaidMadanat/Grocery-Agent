//
//  MacroAnalyticsView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Charts
import SwiftUI

struct MacroAnalyticsView: View {
    @ObservedObject var appModel: AppViewModel
    @StateObject private var viewModel: MacroAnalyticsViewModel
    @State private var selectedMacro: MacroAnalyticsViewModel.MacroType = .calories

    init(appModel: AppViewModel) {
        self.appModel = appModel
        _viewModel = StateObject(wrappedValue: MacroAnalyticsViewModel(appModel: appModel))
    }

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 24) {
                Picker("Macro", selection: $selectedMacro) {
                    ForEach(MacroAnalyticsViewModel.MacroType.allCases, id: \.self) { macro in
                        Text(macro.rawValue).tag(macro)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 24)

                Chart {
                    ForEach(viewModel.data(for: selectedMacro)) { point in
                        BarMark(
                            x: .value("Day", point.day),
                            y: .value(selectedMacro.rawValue, point.value)
                        )
                        .foregroundStyle(by: .value("Metric", selectedMacro.rawValue))
                    }
                }
                .chartLegend(.hidden)
                .frame(height: 280)
                .padding(.horizontal, 24)

                trendCard
                    .padding(.horizontal, 24)

                Spacer()
            }
            .navigationTitle("Macros")
            .background(Color.agentGroupedBackground.ignoresSafeArea())
        }
    }

    private var trendCard: some View {
        let delta = viewModel.trend(for: selectedMacro)
        let symbol = delta >= 0 ? "arrow.up" : "arrow.down"
        let color: Color = delta >= 0 ? .green : .orange

        return VStack(alignment: .leading, spacing: 8) {
            Label("Weekly trend", systemImage: symbol)
                .font(.headline)
                .foregroundStyle(color)
            Text("\(selectedMacro.rawValue) changed by \(Int(delta)) \(selectedMacro.unit) since Monday.")
                .font(.subheadline)
            if selectedMacro == .calories {
                Text("Aim to stay within ±200 kcals of your target for consistent results.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("Tap a bar to inspect daily totals and adjust your plan accordingly.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .shadow(color: Color.black.opacity(0.08), radius: 12, x: 0, y: 6)
    }
}

#Preview {
    MacroAnalyticsView(appModel: AppViewModel())
}

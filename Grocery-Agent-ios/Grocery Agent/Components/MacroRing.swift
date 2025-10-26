//
//  MacroRing.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct MacroRing: View {
    let title: String
    let value: Double
    let goal: Double
    let gradient: Gradient
    let unit: String
    var size: CGFloat = 96

    private var progress: Double {
        guard goal > 0 else { return 0 }
        return min(value / goal, 1.2)
    }

    var body: some View {
        VStack(spacing: 12) {
            ZStack {
                Circle()
                    .stroke(Color.secondary.opacity(0.15), lineWidth: 12)
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(
                        AngularGradient(gradient: gradient, center: .center),
                        style: StrokeStyle(lineWidth: 12, lineCap: .round)
                    )
                    .rotationEffect(.degrees(-90))
                    .animation(.easeOut(duration: 0.6), value: progress)

                VStack {
                    Text("\(Int(value))")
                        .font(.title2.bold())
                    Text(unit)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: size, height: size)

            Text(title)
                .font(.subheadline.bold())
                .frame(width: size)
        }
    }
}

#Preview {
    MacroRing(
        title: "Protein",
        value: 98,
        goal: 120,
        gradient: Gradient(colors: [.orange, .pink]),
        unit: "g"
    )
    .padding()
}

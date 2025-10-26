//
//  FloatingActionButton.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct FloatingActionButton: View {
    var icon: String
    var title: String
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.title3.bold())
                Text(title)
                    .font(.headline)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .background(Material.ultraThinMaterial, in: Capsule(style: .circular))
            .shadow(color: Color.black.opacity(0.15), radius: 20, x: 0, y: 12)
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    FloatingActionButton(icon: "sparkles", title: "Regenerate", action: {})
        .padding()
        .background(Color.agentGroupedBackground)
}

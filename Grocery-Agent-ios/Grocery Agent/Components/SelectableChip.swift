//
//  SelectableChip.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct SelectableChip: View {
    let title: String
    var systemImage: String?
    var isSelected: Bool
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .symbolRenderingMode(.palette)
                        .foregroundStyle(isSelected ? Color.white : Color.accentColor)
                        .frame(width: 18, height: 18)
                }
                Text(title)
                    .font(.callout.weight(.semibold))
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, minHeight: 52)
            .padding(.vertical, 4)
            .background(background)
            .overlay(
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .strokeBorder(borderColor, lineWidth: isSelected ? 1.5 : 0)
            )
            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            .contentShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            .foregroundStyle(isSelected ? Color.white : Color.accentColor)
        }
        .buttonStyle(.plain)
        .animation(.spring(response: 0.25, dampingFraction: 0.75), value: isSelected)
    }

    private var background: AnyShapeStyle {
        if isSelected {
            AnyShapeStyle(LinearGradient(colors: [.blue, .green], startPoint: .leading, endPoint: .trailing))
        } else {
            AnyShapeStyle(Color.clear)
        }
    }

    private var borderColor: Color {
        isSelected ? Color.white.opacity(0.45) : Color.clear
    }
}

#Preview {
    VStack(spacing: 16) {
        SelectableChip(title: "Vegetarian", systemImage: "leaf", isSelected: true, action: {})
        SelectableChip(title: "Other", systemImage: "square.and.pencil", isSelected: false, action: {})
    }
    .padding()
    .background(Color.agentGroupedBackground)
}

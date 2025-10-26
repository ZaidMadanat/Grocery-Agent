//
//  InventoryView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct InventoryView: View {
    @ObservedObject var appModel: AppViewModel
    @StateObject private var viewModel: InventoryViewModel

    init(appModel: AppViewModel) {
        self.appModel = appModel
        _viewModel = StateObject(wrappedValue: InventoryViewModel(appModel: appModel))
    }

    var body: some View {
        NavigationStack {
            List {
                ForEach(viewModel.groupedByCategory, id: \.key) { category, items in
                    Section(category) {
                        ForEach(items) { item in
                            InventoryRow(item: item, descriptor: viewModel.color(for: item.statusColor)) { newAmount in
                                viewModel.updateQuantity(for: item, amount: newAmount)
                            } onRemove: {
                                viewModel.remove(item)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Inventory")
        }
    }
}

private struct InventoryRow: View {
    let item: InventoryItem
    let descriptor: InventoryViewModel.ColorDescriptor
    var onUpdate: (Int) -> Void
    var onRemove: () -> Void
    @State private var amount: Int

    init(
        item: InventoryItem,
        descriptor: InventoryViewModel.ColorDescriptor,
        onUpdate: @escaping (Int) -> Void,
        onRemove: @escaping () -> Void
    ) {
        self.item = item
        self.descriptor = descriptor
        self.onUpdate = onUpdate
        self.onRemove = onRemove
        let numeric = Int(item.quantity.filter { $0.isNumber }) ?? 1
        _amount = State(initialValue: max(numeric, 0))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(item.name)
                    .font(.headline)
                Spacer()
                Text(descriptor.label)
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(descriptor.color.opacity(0.15), in: Capsule())
            }

            HStack(alignment: .center, spacing: 16) {
                Label("\(item.daysUntilExpiry) days left", systemImage: "clock")
                    .foregroundStyle(descriptor.color)
                Spacer()
                QuantityStepper(amount: $amount, color: descriptor.color) {
                    onUpdate(amount)
                }
                Button(role: .destructive, action: onRemove) {
                    Image(systemName: "trash")
                }
                .buttonStyle(.borderless)
            }
            if item.daysUntilExpiry <= 2 {
                Label("Notification scheduled", systemImage: "bell.badge")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}

private struct QuantityStepper: View {
    @Binding var amount: Int
    var color: Color
    var onUpdate: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Button {
                amount = max(amount - 1, 0)
                onUpdate()
            } label: {
                Image(systemName: "minus")
                    .font(.subheadline.weight(.bold))
                    .frame(width: 30, height: 30)
                    .background(color.opacity(0.15), in: Circle())
            }
            .buttonStyle(.plain)

            Text("\(amount)")
                .font(.headline)
                .frame(minWidth: 32)

            Button {
                amount += 1
                onUpdate()
            } label: {
                Image(systemName: "plus")
                    .font(.subheadline.weight(.bold))
                    .frame(width: 30, height: 30)
                    .background(color.opacity(0.15), in: Circle())
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 6)
    }
}

#Preview {
    InventoryView(appModel: AppViewModel())
}

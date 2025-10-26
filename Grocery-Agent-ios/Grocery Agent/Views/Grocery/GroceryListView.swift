//
//  GroceryListView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct GroceryListView: View {
    @ObservedObject var appModel: AppViewModel
    @StateObject private var viewModel: GroceryListViewModel
    @State private var presentRemoval: GroceryItem?

    init(appModel: AppViewModel) {
        self.appModel = appModel
        _viewModel = StateObject(wrappedValue: GroceryListViewModel(appModel: appModel))
    }

    var body: some View {
        NavigationStack {
            List {
                summaryHeader

                ForEach(viewModel.groupedByDay, id: \.key) { day, items in
                    Section(day) {
                        ForEach(items) { item in
                            HStack(spacing: 12) {
                                Button {
                                    viewModel.toggleCompletion(for: item)
                                } label: {
                                    Image(systemName: item.isChecked ? "checkmark.circle.fill" : "circle")
                                        .foregroundStyle(item.isChecked ? .green : .secondary)
                                }
                                .buttonStyle(.plain)

                                VStack(alignment: .leading, spacing: 4) {
                                    Text(item.name)
                                        .strikethrough(item.isChecked)
                                    Text("\(item.quantity) • \(item.mealType.rawValue)")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }

                                Spacer()

                                QuantityEditor(
                                    quantity: item.quantity,
                                    onCommit: { newValue in
                                        viewModel.updateQuantity(newValue, for: item)
                                    }
                                )

                                Button(role: .destructive) {
                                    presentRemoval = item
                                } label: {
                                    Image(systemName: "trash")
                                }
                                .buttonStyle(.borderless)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }

                checkoutFooter
            }
            .navigationTitle("Grocery List")
            .alert("Remove item?", isPresented: Binding(
                get: { presentRemoval != nil },
                set: { if !$0 { presentRemoval = nil } }
            )) {
                Button("Delete", role: .destructive) {
                    if let item = presentRemoval {
                        viewModel.remove(item)
                    }
                    presentRemoval = nil
                }
                Button("Cancel", role: .cancel) { presentRemoval = nil }
            } message: {
                Text("We'll regenerate this ingredient next sync if it's still needed.")
            }
        }
    }

    private var summaryHeader: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("You have \(viewModel.totalItems) items")
                .font(.headline)
            Text("Tap to mark items purchased or remove them if you already have them on hand.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .listRowInsets(EdgeInsets())
    }

    private var checkoutFooter: some View {
        Section {
            Link(destination: viewModel.checkoutURL) {
                HStack {
                    Image(systemName: "cart.badge.plus")
                    Text("Open Kroger / Instacart checkout")
                }
                .frame(maxWidth: .infinity, alignment: .center)
            }
            .buttonStyle(.borderedProminent)
        } footer: {
            Text("We'll route to Kroger when available otherwise fall back to Instacart deep link.")
        }
    }
}

#Preview {
    GroceryListView(appModel: AppViewModel())
}

private struct QuantityEditor: View {
    @State private var text: String
    var onCommit: (String) -> Void

    init(quantity: String, onCommit: @escaping (String) -> Void) {
        _text = State(initialValue: quantity)
        self.onCommit = onCommit
    }

    var body: some View {
        TextField("Qty", text: $text)
            .textFieldStyle(.roundedBorder)
            .frame(width: 90)
            .multilineTextAlignment(.center)
            .onSubmit { commit() }
            .onChange(of: text) { newValue in
                commit()
            }
    }

    private func commit() {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        onCommit(trimmed)
    }
}

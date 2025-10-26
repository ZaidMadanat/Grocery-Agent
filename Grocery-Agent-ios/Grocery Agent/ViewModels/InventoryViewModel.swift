//
//  InventoryViewModel.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Combine
import Foundation
import SwiftUI

@MainActor
final class InventoryViewModel: ObservableObject {
    @Published private(set) var items: [InventoryItem] = []

    private let appModel: AppViewModel

    init(appModel: AppViewModel) {
        self.appModel = appModel
        items = appModel.inventoryItems
    }

    var groupedByCategory: [(key: String, value: [InventoryItem])] {
        Dictionary(grouping: items) { $0.category }
            .sorted { $0.key < $1.key }
    }

    func color(for status: InventoryStatus) -> ColorDescriptor {
        switch status {
        case .fresh: return ColorDescriptor(color: .green, label: "Fresh")
        case .warning: return ColorDescriptor(color: .yellow, label: "3-5 days")
        case .urgent: return ColorDescriptor(color: .red, label: "Expiring soon")
        case .expired: return ColorDescriptor(color: .gray, label: "Expired")
        }
    }

    func updateQuantity(for item: InventoryItem, amount: Int) {
        guard let index = items.firstIndex(where: { $0.id == item.id }) else { return }
        var updated = items[index]
        updated.quantity = "\(amount)"
        items[index] = updated
        appModel.updateInventoryItem(updated, amount: amount)
    }

    func remove(_ item: InventoryItem) {
        items.removeAll { $0.id == item.id }
    }

    struct ColorDescriptor {
        let color: Color
        let label: String
    }
}

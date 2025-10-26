//
//  GroceryListViewModel.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Combine
import Foundation

@MainActor
final class GroceryListViewModel: ObservableObject {
    @Published private(set) var items: [GroceryItem] = []
    @Published var showingRemovalConfirmation: GroceryItem?

    private let appModel: AppViewModel

    init(appModel: AppViewModel) {
        self.appModel = appModel
        items = appModel.groceryItems
    }

    var totalItems: Int {
        items.count
    }

    var checkoutURL: URL {
        MockDataService.checkoutURL()
    }

    var groupedByDay: [(key: String, value: [GroceryItem])] {
        Dictionary(grouping: items) { $0.dayLabel }
            .sorted { $0.key < $1.key }
    }

    func toggleCompletion(for item: GroceryItem) {
        appModel.toggleGroceryItem(item)
        items = appModel.groceryItems
    }

    func remove(_ item: GroceryItem) {
        appModel.removeGroceryItem(item)
        items = appModel.groceryItems
    }

    func updateQuantity(_ quantity: String, for item: GroceryItem) {
        appModel.updateGroceryItem(item, quantity: quantity)
        items = appModel.groceryItems
    }
}

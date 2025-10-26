//
//  MainShellView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct MainShellView: View {
    @ObservedObject var appModel: AppViewModel

    var body: some View {
        TabView {
            DashboardView(appModel: appModel)
                .tabItem {
                    Label("Plan", systemImage: "calendar")
                }

            MacroAnalyticsView(appModel: appModel)
                .tabItem {
                    Label("Macros", systemImage: "chart.pie")
                }

            GroceryListView(appModel: appModel)
                .tabItem {
                    Label("Groceries", systemImage: "cart")
                }

            InventoryView(appModel: appModel)
                .tabItem {
                    Label("Inventory", systemImage: "shippingbox")
                }

            NotificationsView(appModel: appModel)
                .tabItem {
                    Label("Alerts", systemImage: "bell")
                }

            SettingsView(appModel: appModel)
                .tabItem {
                    Label("Settings", systemImage: "gearshape")
                }
        }
        .tint(.accentColor)
    }
}

#Preview {
    MainShellView(appModel: AppViewModel())
}

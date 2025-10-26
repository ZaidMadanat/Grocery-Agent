//
//  NotificationsView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct NotificationsView: View {
    @ObservedObject var appModel: AppViewModel
    @StateObject private var viewModel: NotificationsViewModel

    init(appModel: AppViewModel) {
        self.appModel = appModel
        _viewModel = StateObject(wrappedValue: NotificationsViewModel(appModel: appModel))
    }

    var body: some View {
        NavigationStack {
            List {
                if viewModel.notifications.isEmpty {
                    ContentUnavailableView("No notifications yet", systemImage: "bell", description: Text("We'll drop reminders here when inventory or grocery updates occur."))
                } else {
                    ForEach(viewModel.notifications) { notification in
                        NotificationRow(notification: notification)
                            .padding(.vertical, 8)
                            .listRowBackground(notification.type.accentColor.opacity(0.05))
                            .onTapGesture {
                                viewModel.markAsRead(notification)
                            }
                    }
                }
            }
            .navigationTitle("Alerts")
        }
    }
}

private struct NotificationRow: View {
    let notification: AgentNotification

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            Image(systemName: notification.type.symbolName)
                .font(.title3)
                .foregroundStyle(notification.type.accentColor)
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(notification.title)
                        .font(.headline)
                    if !notification.isRead {
                        Circle()
                            .fill(Color.blue)
                            .frame(width: 8, height: 8)
                    }
                }
                Text(notification.message)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Text(notification.createdAt, style: .relative)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

#Preview {
    NotificationsView(appModel: AppViewModel())
}

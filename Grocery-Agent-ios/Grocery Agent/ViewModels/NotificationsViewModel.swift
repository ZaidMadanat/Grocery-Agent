//
//  NotificationsViewModel.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Combine
import Foundation

@MainActor
final class NotificationsViewModel: ObservableObject {
    @Published private(set) var notifications: [AgentNotification] = []

    private let appModel: AppViewModel

    init(appModel: AppViewModel) {
        self.appModel = appModel
        notifications = appModel.notifications
    }

    func markAsRead(_ notification: AgentNotification) {
        appModel.markNotificationRead(notification)
        notifications = appModel.notifications
    }

    func unreadCount() -> Int {
        notifications.filter { !$0.isRead }.count
    }
}

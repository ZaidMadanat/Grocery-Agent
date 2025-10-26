//
//  NotificationModels.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Foundation
import SwiftUI

/// Represents in-app notification or alert surfaced to the user.
struct AgentNotification: Identifiable, Codable {
    enum NotificationType: String, Codable {
        case expiry
        case grocery
        case recipe
        case system

        var symbolName: String {
            switch self {
            case .expiry: return "exclamationmark.triangle.fill"
            case .grocery: return "cart.fill"
            case .recipe: return "book.pages.fill"
            case .system: return "info.circle.fill"
            }
        }

        var accentColor: Color {
            switch self {
            case .expiry: return .orange
            case .grocery: return .green
            case .recipe: return .blue
            case .system: return .gray
            }
        }
    }

    let id: UUID
    var title: String
    var message: String
    var createdAt: Date
    var type: NotificationType
    var isRead: Bool

    init(
        id: UUID = UUID(),
        title: String,
        message: String,
        createdAt: Date = .now,
        type: NotificationType,
        isRead: Bool = false
    ) {
        self.id = id
        self.title = title
        self.message = message
        self.createdAt = createdAt
        self.type = type
        self.isRead = isRead
    }
}

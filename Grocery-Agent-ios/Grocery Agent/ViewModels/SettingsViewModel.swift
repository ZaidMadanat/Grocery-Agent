//
//  SettingsViewModel.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Combine
import Foundation

@MainActor
final class SettingsViewModel: ObservableObject {
    @Published var isVoiceModeEnabled = true
    @Published var syncStatus = "Synced 5m ago"
    @Published var showResetConfirmation = false
    @Published var showLogoutConfirmation = false

    private let appModel: AppViewModel

    init(appModel: AppViewModel) {
        self.appModel = appModel
    }

    func resetPreferences() {
        appModel.resetOnboarding()
    }
}

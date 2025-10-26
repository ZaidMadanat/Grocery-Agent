//
//  SettingsView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct SettingsView: View {
    @ObservedObject var appModel: AppViewModel
    @StateObject private var viewModel: SettingsViewModel

    init(appModel: AppViewModel) {
        self.appModel = appModel
        _viewModel = StateObject(wrappedValue: SettingsViewModel(appModel: appModel))
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Account") {
                    Label("madinat@example.com", systemImage: "envelope.fill")
                    Label("Connected to Letta Cloud", systemImage: "network")
                    HStack {
                        Label("Sync Status", systemImage: "arrow.clockwise")
                        Spacer()
                        Text(viewModel.syncStatus)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Voice & Notifications") {
                    Toggle(isOn: $viewModel.isVoiceModeEnabled) {
                        Label("Voice assistant mode", systemImage: "waveform.circle")
                    }
                    Label("Expiry notifications enabled", systemImage: "bell.badge")
                }

                Section("Actions") {
                    Button(role: .destructive) {
                        viewModel.showResetConfirmation = true
                    } label: {
                        Label("Reset preferences", systemImage: "arrow.uturn.backward")
                    }

                    Button(role: .destructive) {
                        viewModel.showLogoutConfirmation = true
                    } label: {
                        Label("Log out", systemImage: "rectangle.portrait.and.arrow.right")
                    }
                }
            }
            .navigationTitle("Settings")
            .alert("Reset Preferences?", isPresented: $viewModel.showResetConfirmation) {
                Button("Reset", role: .destructive) {
                    viewModel.resetPreferences()
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("You'll return to onboarding and need to resubmit your dietary selections.")
            }
            .alert("Log Out?", isPresented: $viewModel.showLogoutConfirmation) {
                Button("Log Out", role: .destructive) {
                    // Placeholder for logout flow.
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("We'll clear session tokens and require re-authentication.")
            }
        }
    }
}

#Preview {
    SettingsView(appModel: AppViewModel())
}

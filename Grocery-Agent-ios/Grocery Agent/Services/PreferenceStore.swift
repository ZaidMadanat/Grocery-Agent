//
//  PreferenceStore.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Foundation

/// Lightweight persistence wrapper for user preferences.
@MainActor
final class PreferenceStore {
    private enum Constants {
        static let storageKey = "agent.user.preferences"
    }

    private let defaults: UserDefaults
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    func loadPreferences() -> UserPreferences? {
        guard let data = defaults.data(forKey: Constants.storageKey) else {
            return nil
        }

        do {
            return try decoder.decode(UserPreferences.self, from: data)
        } catch {
            print("Failed to decode preferences: \(error)")
            return nil
        }
    }

    func save(_ preferences: UserPreferences) {
        do {
            let data = try encoder.encode(preferences)
            defaults.set(data, forKey: Constants.storageKey)
        } catch {
            print("Failed to encode preferences: \(error)")
        }
    }

    func reset() {
        defaults.removeObject(forKey: Constants.storageKey)
    }
}

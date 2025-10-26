import Foundation

final class AuthService {
    static let shared = AuthService()
    private let key = "agent.jwt"

    func setToken(_ token: String) {
        UserDefaults.standard.set(token, forKey: key)
    }

    func getToken() -> String? {
        UserDefaults.standard.string(forKey: key)
    }
}


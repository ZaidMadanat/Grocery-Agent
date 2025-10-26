import Foundation

enum AppEnvironment {
    static let apiBaseURL: URL = {
        if let override = ProcessInfo.processInfo.environment["API_BASE_URL"],
           let url = URL(string: override) {
            return url
        }
        return URL(string: "http://localhost:8000")!
    }()
}

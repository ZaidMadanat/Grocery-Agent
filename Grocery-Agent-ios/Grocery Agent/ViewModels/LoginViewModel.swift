import Foundation
import Combine

@MainActor
final class LoginViewModel: ObservableObject {
    @Published var email: String = "you@example.com"
    @Published var password: String = "passw0rd!"
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    func login(completion: @escaping () -> Void) {
        guard !isLoading else { return }
        isLoading = true
        errorMessage = nil

        Task {
            do {
                let client = APIClient(tokenProvider: { nil })
                let resp: TokenResponse = try await client.postJSON(
                    "/auth/login",
                    payload: LoginRequest(email: email, password: password)
                )
                AuthService.shared.setToken(resp.access_token)
                isLoading = false
                completion()
            } catch {
                isLoading = false
                errorMessage = "Login failed. Check your email/password and API."
            }
        }
    }
}

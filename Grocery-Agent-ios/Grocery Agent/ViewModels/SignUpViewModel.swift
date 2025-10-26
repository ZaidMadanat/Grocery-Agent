import Foundation
import Combine

@MainActor
final class SignUpViewModel: ObservableObject {
    @Published var name: String = ""
    @Published var username: String = ""
    @Published var email: String = ""
    @Published var password: String = ""
    @Published var confirmPassword: String = ""
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?

    var isFormValid: Bool {
        !name.isEmpty &&
        !username.isEmpty &&
        !email.isEmpty &&
        !password.isEmpty &&
        password == confirmPassword &&
        password.count >= 8 &&
        email.contains("@")
    }

    func signUp(completion: @escaping () -> Void) {
        guard !isLoading else { return }
        guard isFormValid else {
            errorMessage = "Please fill all fields correctly."
            return
        }

        isLoading = true
        errorMessage = nil

        Task {
            do {
                let client = APIClient(tokenProvider: { nil })
                let resp: TokenResponse = try await client.postJSON(
                    "/auth/register",
                    payload: RegisterRequest(
                        email: email,
                        username: username,
                        password: password,
                        name: name,
                        daily_calories: 2200,
                        dietary_restrictions: [],
                        likes: []
                    )
                )
                AuthService.shared.setToken(resp.access_token)
                isLoading = false
                completion()
            } catch {
                isLoading = false
                errorMessage = "Registration failed. Email or username may already exist."
            }
        }
    }
}


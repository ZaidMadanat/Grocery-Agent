import Foundation

struct LoginRequest: Encodable {
    let email: String
    let password: String
}

struct TokenResponse: Decodable {
    let access_token: String
    let token_type: String
    let expires_in: Int
    let user: UserInfo

    struct UserInfo: Decodable {
        let id: Int
        let email: String
        let username: String
        let name: String
    }
}


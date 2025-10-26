import Foundation

struct SessionCreateRequest: Encodable {
    let room: String
    let ttl_seconds: Int
}

struct SessionCreateResponse: Decodable {
    let token: String
    let url: String
    let room: String
    let identity: String
}

enum VoiceService {
    static func createSession() async throws -> SessionCreateResponse {
        let client = APIClient(tokenProvider: { AuthService.shared.getToken() })
        let req = SessionCreateRequest(room: "cooking-demo", ttl_seconds: 3600)
        let resp: SessionCreateResponse = try await client.postJSON("/session/create", payload: req)
        return resp
    }
}


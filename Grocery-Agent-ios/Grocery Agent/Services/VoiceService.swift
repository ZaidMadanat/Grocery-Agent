import Foundation

struct SessionCreateRequest: Encodable {
    let room: String?
    let identity: String?
    let ttlSeconds: Int

    enum CodingKeys: String, CodingKey {
        case room
        case identity
        case ttlSeconds = "ttl_seconds"
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        if let room {
            try container.encode(room, forKey: .room)
        }
        if let identity {
            try container.encode(identity, forKey: .identity)
        }
        try container.encode(ttlSeconds, forKey: .ttlSeconds)
    }
}

struct SessionCreateResponse: Decodable {
    let token: String
    let url: String
    let room: String
    let identity: String
    let expiresAt: Date

    private enum CodingKeys: String, CodingKey {
        case token
        case url
        case room
        case identity
        case expiresAt = "expires_at"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        token = try container.decode(String.self, forKey: .token)
        url = try container.decode(String.self, forKey: .url)
        room = try container.decode(String.self, forKey: .room)
        identity = try container.decode(String.self, forKey: .identity)

        let expiresRaw = try container.decode(String.self, forKey: .expiresAt)
        guard let parsedDate = SessionCreateResponse.isoFormatter.date(from: expiresRaw) else {
            throw DecodingError.dataCorruptedError(
                forKey: .expiresAt,
                in: container,
                debugDescription: "Unable to parse expires_at '\(expiresRaw)'"
            )
        }
        expiresAt = parsedDate
    }

    private static let isoFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
}

enum VoiceService {
    static func createSession(room: String? = nil, identity: String? = nil, ttlSeconds: Int = 3600) async throws -> SessionCreateResponse {
        let client = APIClient(tokenProvider: { AuthService.shared.getToken() })
        let request = SessionCreateRequest(room: room, identity: identity, ttlSeconds: ttlSeconds)
        return try await client.postJSON("/session/create", payload: request)
    }
}

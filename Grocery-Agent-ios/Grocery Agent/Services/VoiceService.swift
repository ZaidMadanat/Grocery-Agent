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
    let expiresAt: Date?

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

        if let expiresRaw = try container.decodeIfPresent(String.self, forKey: .expiresAt) {
            expiresAt = SessionCreateResponse.parseDate(from: expiresRaw)
        } else {
            expiresAt = nil
        }
    }

    private static func parseDate(from raw: String) -> Date? {
        for formatter in iso8601Formatters {
            if let date = formatter.date(from: raw) {
                return date
            }
        }
        return nil
    }

    private static let iso8601Formatters: [ISO8601DateFormatter] = {
        let withFractional = ISO8601DateFormatter()
        withFractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        return [withFractional, plain]
    }()

    var liveKitURL: URL? {
        URL(string: url)
    }
}

enum VoiceService {
    static func createSession(room: String? = nil, identity: String? = nil, ttlSeconds: Int = 3600) async throws -> SessionCreateResponse {
        let client = APIClient(tokenProvider: { AuthService.shared.getToken() })
        let request = SessionCreateRequest(room: room, identity: identity, ttlSeconds: ttlSeconds)
        return try await client.postJSON("/session/create", payload: request)
    }
}

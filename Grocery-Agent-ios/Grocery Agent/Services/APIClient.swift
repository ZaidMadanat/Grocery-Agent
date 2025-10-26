import Foundation

struct APIClient {
    let baseURL: URL
    let tokenProvider: () -> String?

    private func request(_ path: String,
                         method: String = "GET",
                         query: [URLQueryItem]? = nil,
                         body: Data? = nil) -> URLRequest {
        var url = baseURL.appendingPathComponent(path)
        if let query = query, var comps = URLComponents(url: url, resolvingAgainstBaseURL: false) {
            comps.queryItems = query
            url = comps.url ?? url
        }
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let t = tokenProvider() {
            req.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization")
        }
        req.httpBody = body
        return req
    }

    func postJSON<T: Decodable>(_ path: String,
                                payload: Encodable?,
                                query: [URLQueryItem]? = nil) async throws -> T {
        let data = try payload.map { try JSONEncoder().encode(AnyEncodable($0)) }
        let req = request(path, method: "POST", query: query, body: data)
        let (respData, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(T.self, from: respData)
    }

    // Wrapper to encode any Encodable
    private struct AnyEncodable: Encodable {
        let value: Encodable
        init(_ value: Encodable) { self.value = value }
        func encode(to encoder: Encoder) throws { try value.encode(to: encoder) }
    }
}


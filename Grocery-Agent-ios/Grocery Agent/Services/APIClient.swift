import Foundation

struct APIClient {
    let baseURL: URL
    let tokenProvider: () -> String?

    init(baseURL: URL = AppEnvironment.apiBaseURL,
         tokenProvider: @escaping () -> String?) {
        self.baseURL = baseURL
        self.tokenProvider = tokenProvider
    }

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
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        if let t = tokenProvider() {
            req.setValue("Bearer \(t)", forHTTPHeaderField: "Authorization")
        }
        req.httpBody = body
        return req
    }

    func getJSON<T: Decodable>(_ path: String,
                               query: [URLQueryItem]? = nil) async throws -> T {
        let req = request(path, method: "GET", query: query, body: nil)
        return try await perform(req)
    }

    func postJSON<T: Decodable>(_ path: String,
                                payload: Encodable?,
                                query: [URLQueryItem]? = nil) async throws -> T {
        let data = try payload.map { try JSONEncoder().encode(AnyEncodable($0)) }
        let req = request(path, method: "POST", query: query, body: data)
        return try await perform(req)
    }

    // Wrapper to encode any Encodable
    private struct AnyEncodable: Encodable {
        let value: Encodable
        init(_ value: Encodable) { self.value = value }
        func encode(to encoder: Encoder) throws { try value.encode(to: encoder) }
    }

    private func perform<T: Decodable>(_ request: URLRequest) async throws -> T {
        let (respData, resp) = try await URLSession.shared.data(for: request)
        guard let http = resp as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200..<300).contains(http.statusCode) else {
            if let debugBody = String(data: respData, encoding: .utf8) {
                print("[APIClient] \(http.statusCode) error: \(debugBody)")
            }
            throw URLError(.init(rawValue: http.statusCode))
        }
        return try JSONDecoder().decode(T.self, from: respData)
    }
}

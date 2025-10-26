import Foundation
import Combine
import LiveKit

@MainActor
final class VoiceSessionManager: ObservableObject {
    static let shared = VoiceSessionManager()

    @Published private(set) var isConnecting = false
    @Published private(set) var isConnected = false
    @Published private(set) var activeRoomName: String?
    @Published private(set) var lastError: Error?

    private var room: Room?

    private init() {}

    func startSession() async {
        guard !isConnecting else { return }

        isConnecting = true
        lastError = nil

        do {
            let session = try await VoiceService.createSession()

            let room = Room()
            room.add(delegate: self)

            try await room.connect(url: session.url, token: session.token)

            if let localParticipant = room.localParticipant {
                try await localParticipant.setCamera(enabled: false)
                try await localParticipant.setMicrophone(enabled: true)
            }

            self.room = room
            activeRoomName = session.room
            isConnected = true
        } catch {
            lastError = error
            await cleanup()
        }

        isConnecting = false
    }

    func stopSession() async {
        await cleanup()
    }

    func toggleSession() async {
        if isConnected || isConnecting {
            await stopSession()
        } else {
            await startSession()
        }
    }

    private func cleanup() async {
        if let room = room {
            room.remove(delegate: self)
            try? await room.disconnect()
        }

        room = nil
        activeRoomName = nil
        isConnected = false
        isConnecting = false
    }
}

extension VoiceSessionManager: RoomDelegate {
    nonisolated func room(_ room: Room, didDisconnectWith error: LiveKitError?) {
        Task { @MainActor in
            if let error {
                self.lastError = error
            }
            await self.cleanup()
        }
    }

    nonisolated func room(_ room: Room, participantDidConnect participant: RemoteParticipant) {}
    nonisolated func room(_ room: Room, participantDidDisconnect participant: RemoteParticipant) {}
    nonisolated func room(_ room: Room, trackPublication publication: RemoteTrackPublication, didSubscribe track: Track) {}
    nonisolated func room(_ room: Room, trackPublication publication: RemoteTrackPublication, didUnsubscribe track: Track?) {}
}

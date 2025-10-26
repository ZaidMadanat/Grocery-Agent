import Foundation
import Combine
import LiveKit
import AVFoundation

@MainActor
final class VoiceSessionManager: ObservableObject {
    static let shared = VoiceSessionManager()

    @Published private(set) var isConnecting = false
    @Published private(set) var isConnected = false
    @Published private(set) var activeRoomName: String?
    @Published private(set) var lastError: Error?

    private var room: Room?
    private var audioSessionConfigured = false

    private init() {}

    func startSession() async {
        guard !isConnecting else { return }

        isConnecting = true
        lastError = nil

        do {
            try configureAudioSession()
            try AVAudioSession.sharedInstance().setActive(true, options: [])

            let session = try await VoiceService.createSession(room: activeRoomName)

            let room = Room()
            room.add(delegate: self)

            try await room.connect(url: session.url, token: session.token)

            try await room.localParticipant.setCamera(enabled: false)
            try await room.localParticipant.setMicrophone(enabled: true)

            self.room = room
            activeRoomName = session.room
            isConnected = true
        } catch {
            lastError = error
            await cleanup()
        }

        isConnecting = false
    }

    private func configureAudioSession() throws {
        let audioSession = AVAudioSession.sharedInstance()
        let desiredOptions: AVAudioSession.CategoryOptions = [.defaultToSpeaker, .allowBluetooth, .allowBluetoothA2DP]

        try audioSession.setCategory(.playAndRecord, mode: .voiceChat, options: desiredOptions)
        try audioSession.setPreferredSampleRate(48_000)
        try audioSession.setPreferredIOBufferDuration(0.01)
        try audioSession.setActive(true, options: [])

        audioSessionConfigured = true
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

    private func cleanup(deactivateAudio: Bool = true) async {
        if let room = room {
            room.remove(delegate: self)
            try? await room.disconnect()
        }

        if deactivateAudio, audioSessionConfigured {
            let audioSession = AVAudioSession.sharedInstance()
            try? audioSession.setCategory(.soloAmbient, mode: .default, options: [])
            try? audioSession.setActive(false, options: [.notifyOthersOnDeactivation])
            audioSessionConfigured = false
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

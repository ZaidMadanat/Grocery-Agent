//
//  ContentView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var appModel = AppViewModel()
    @State private var hasToken: Bool = AuthService.shared.getToken() != nil

    var body: some View {
        Group {
            if !hasToken {
                LoginView {
                    hasToken = true
                }
                .transition(.opacity)
            } else if appModel.isOnboarded {
                MainShellView(appModel: appModel)
                    .transition(.opacity.combined(with: .scale))
            } else {
                OnboardingFlowView(appModel: appModel)
                    .transition(.move(edge: .trailing))
            }
        }
        .animation(.easeInOut, value: hasToken)
        .animation(.easeInOut, value: appModel.isOnboarded)
    }
}

#Preview {
    ContentView()
}

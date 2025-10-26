//
//  ContentView.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var appModel = AppViewModel()

    var body: some View {
        Group {
            if appModel.isOnboarded {
                MainShellView(appModel: appModel)
                    .transition(.opacity.combined(with: .scale))
            } else {
                OnboardingFlowView(appModel: appModel)
                    .transition(.move(edge: .trailing))
            }
        }
        .animation(.easeInOut, value: appModel.isOnboarded)
    }
}

#Preview {
    ContentView()
}

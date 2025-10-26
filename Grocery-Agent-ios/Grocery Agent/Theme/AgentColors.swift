//
//  AgentColors.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import SwiftUI

#if canImport(UIKit)
import UIKit
#endif

#if canImport(AppKit)
import AppKit
#endif

extension Color {
    static var agentGroupedBackground: Color {
        #if canImport(UIKit)
        Color(uiColor: UIColor.systemGroupedBackground)
        #elseif canImport(AppKit)
        Color(nsColor: NSColor.windowBackgroundColor)
        #else
        Color.gray.opacity(0.12)
        #endif
    }

    static var agentSecondaryBackground: Color {
        #if canImport(UIKit)
        Color(uiColor: UIColor.secondarySystemBackground)
        #elseif canImport(AppKit)
        Color(nsColor: NSColor.underPageBackgroundColor)
        #else
        Color.gray.opacity(0.18)
        #endif
    }

    static var agentSurfaceBackground: Color {
        #if canImport(UIKit)
        Color(uiColor: UIColor.systemBackground)
        #elseif canImport(AppKit)
        Color(nsColor: NSColor.windowBackgroundColor)
        #else
        Color.white
        #endif
    }
}

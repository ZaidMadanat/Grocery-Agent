//
//  DateFormatter+Extensions.swift
//  Grocery Agent
//
//  Created by Zaid Madanat on 10/26/25.
//

import Foundation

extension DateFormatter {
    static let weekdayFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEE"
        return formatter
    }()
}

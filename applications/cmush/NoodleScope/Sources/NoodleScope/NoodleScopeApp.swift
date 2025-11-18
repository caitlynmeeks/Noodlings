import SwiftUI
import Charts

@main
struct NoodleScopeApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .defaultSize(width: 1400, height: 900)
    }
}

struct ContentView: View {
    @StateObject private var viewModel = NoodleScopeViewModel()

    var body: some View {
        VStack(spacing: 0) {
            // Header with scrub controller (Logic Pro style!)
            HeaderView(viewModel: viewModel)
                .frame(height: 60)
                .background(Color(hex: "#131824"))

            // MAIN TIMELINE AREA (Audacity/Logic Pro style)
            MultiTrackTimelineView(viewModel: viewModel)
                .background(Color(hex: "#0a0e1a"))

            // Bottom console for inspecting clicked events
            InspectorConsolePanel(viewModel: viewModel)
                .frame(height: 250)
                .background(Color(hex: "#131824"))
        }
        .background(Color(hex: "#0a0e1a"))
        .task {
            await viewModel.loadSessions()
        }
    }
}

// MARK: - Header View
struct HeaderView: View {
    @ObservedObject var viewModel: NoodleScopeViewModel

    var body: some View {
        HStack {
            Text("NoodleScope 2.0 - Phenomenal State Timeline")
                .font(.system(size: 14, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: "#64b5f6"))

            Spacer()

            // Noodling tabs
            HStack(spacing: 4) {
                ForEach(viewModel.agents, id: \.self) { noodlingId in
                    Button(action: {
                        viewModel.selectedAgent = noodlingId
                    }) {
                        Text(noodlingId.replacingOccurrences(of: "agent_", with: ""))
                            .font(.system(size: 14, design: .monospaced))
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(viewModel.selectedAgent == noodlingId ?
                                       Color(hex: "#2a5f8f") : Color(hex: "#1e2938"))
                            .foregroundColor(.white)
                            .cornerRadius(4)
                    }
                    .buttonStyle(.plain)
                }
            }

            Spacer()

            // Session selector
            Picker("Session:", selection: $viewModel.selectedSession) {
                ForEach(viewModel.sessions, id: \.self) { session in
                    Text(session).tag(session)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 300)

            Button("Refresh") {
                Task { await viewModel.loadSessions() }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(.horizontal, 16)
    }
}

// MARK: - Multi-Track Timeline (Logic Pro / Audacity style!)
struct MultiTrackTimelineView: View {
    @ObservedObject var viewModel: NoodleScopeViewModel
    @State private var expandedTracks: Set<String> = []

    var body: some View {
        VStack(spacing: 0) {
            // SCRUB CONTROLLER (top bar like Logic Pro)
            ScrubController(viewModel: viewModel)
                .frame(height: 60)
                .background(Color(hex: "#131824"))
                .border(Color(hex: "#2a3f5f"), width: 1)

            // TRACK LIST (scrollable like Audacity)
            ScrollView {
                VStack(spacing: 2) {
                    ForEach(viewModel.agents, id: \.self) { noodlingId in
                        NoodlingTrack(
                            noodlingId: noodlingId,
                            viewModel: viewModel,
                            isExpanded: expandedTracks.contains(noodlingId),
                            onToggle: {
                                if expandedTracks.contains(noodlingId) {
                                    expandedTracks.remove(noodlingId)
                                } else {
                                    expandedTracks.insert(noodlingId)
                                }
                            }
                        )
                    }
                }
            }
        }
    }
}

// MARK: - Scrub Controller (playhead with timecode)
struct ScrubController: View {
    @ObservedObject var viewModel: NoodleScopeViewModel

    var body: some View {
        VStack(spacing: 8) {
            // Timecode display
            HStack {
                Text(String(format: "%02d:%02d.%01d",
                           Int(viewModel.playheadTime) / 60,
                           Int(viewModel.playheadTime) % 60,
                           Int((viewModel.playheadTime.truncatingRemainder(dividingBy: 1)) * 10)))
                    .font(.system(size: 20, weight: .bold, design: .monospaced))
                    .foregroundColor(Color(hex: "#64b5f6"))

                Text("/")
                    .foregroundColor(.gray)

                Text(String(format: "%02d:%02d",
                           Int(viewModel.maxTime) / 60,
                           Int(viewModel.maxTime) % 60))
                    .font(.system(size: 16, design: .monospaced))
                    .foregroundColor(.gray)
            }

            // Playhead slider
            Slider(value: $viewModel.playheadTime,
                   in: 0...max(viewModel.maxTime, 0.1),
                   step: 0.1)
                .tint(Color(hex: "#64b5f6"))
                .padding(.horizontal, 12)
        }
        .padding(.horizontal, 16)
    }
}

// MARK: - Noodling Track (collapsible per-Noodling track)
struct NoodlingTrack: View {
    let noodlingId: String
    @ObservedObject var viewModel: NoodleScopeViewModel
    let isExpanded: Bool
    let onToggle: () -> Void

    var noodlingName: String {
        noodlingId.replacingOccurrences(of: "agent_", with: "").capitalized
    }

    var noodlingData: [TimelinePoint] {
        viewModel.timelineData.filter { _ in viewModel.selectedAgent == noodlingId }
    }

    var body: some View {
        VStack(spacing: 0) {
            // TRACK HEADER (like Logic Pro track name)
            Button(action: onToggle) {
                HStack {
                    // Expand/collapse indicator
                    Text(isExpanded ? "▼" : "▶")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.gray)
                        .frame(width: 20)

                    // Noodling name
                    Text(noodlingName.uppercased())
                        .font(.system(size: 12, weight: .bold, design: .monospaced))
                        .foregroundColor(viewModel.selectedAgent == noodlingId ? Color(hex: "#64b5f6") : .white)

                    // Enlightenment indicator (no emoji - just text!)
                    if viewModel.isEnlightened(noodlingId) {
                        Text("[ENLIGHTENED]")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(Color(hex: "#ba68c8"))
                    }

                    Spacer()

                    // Track controls
                    Text("5-D AFFECT")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.gray)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(viewModel.selectedAgent == noodlingId ?
                           Color(hex: "#1e2938") : Color(hex: "#131824"))
            }
            .buttonStyle(.plain)

            // EXPANDED TRACK CONTENT
            if isExpanded {
                VStack(spacing: 4) {
                    // 5-D Affect mini-tracks
                    AffectMiniTrack(title: "V", color: "#66bb6a", data: noodlingData, keyPath: \.valence, viewModel: viewModel)
                    AffectMiniTrack(title: "A", color: "#ffa726", data: noodlingData, keyPath: \.arousal, viewModel: viewModel)
                    AffectMiniTrack(title: "F", color: "#ef5350", data: noodlingData, keyPath: \.fear, viewModel: viewModel)
                    AffectMiniTrack(title: "So", color: "#9c27b0", data: noodlingData, keyPath: \.sorrow, viewModel: viewModel)
                    AffectMiniTrack(title: "Su", color: "#64b5f6", data: noodlingData, keyPath: \.surprise, viewModel: viewModel, highlightSpikes: true)

                    // EVENT TRAFFIC TIMELINE (LLM calls, intuition, etc.)
                    EventTrafficTimeline(noodlingId: noodlingId, data: noodlingData, viewModel: viewModel)
                }
                .padding(.horizontal, 32)
                .padding(.vertical, 8)
                .background(Color(hex: "#0f1419"))
            }
        }
        .border(Color(hex: "#2a3f5f"), width: 1)
    }
}

// MARK: - Affect Mini Track (compact single-line chart)
struct AffectMiniTrack: View {
    let title: String
    let color: String
    let data: [TimelinePoint]
    let keyPath: KeyPath<TimelinePoint, Double>
    @ObservedObject var viewModel: NoodleScopeViewModel
    var highlightSpikes: Bool = false

    var body: some View {
        HStack(spacing: 8) {
            // Label
            Text(title)
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: color))
                .frame(width: 30, alignment: .trailing)

            // Mini chart
            Chart {
                ForEach(data) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value(title, point[keyPath: keyPath])
                    )
                    .foregroundStyle(Color(hex: color))
                    .lineStyle(StrokeStyle(lineWidth: 1.5))

                    // Highlight spikes
                    if highlightSpikes && point[keyPath: keyPath] > 0.3 {
                        PointMark(
                            x: .value("Time", point.timestamp),
                            y: .value(title, point[keyPath: keyPath])
                        )
                        .foregroundStyle(Color(hex: "#ffeb3b"))
                        .symbol(.circle)
                        .symbolSize(40)
                    }
                }

                // Playhead
                RuleMark(x: .value("Playhead", viewModel.playheadTime))
                    .foregroundStyle(.white.opacity(0.5))
                    .lineStyle(StrokeStyle(lineWidth: 1))
            }
            .chartXScale(domain: 0...max(viewModel.maxTime, 0.1))
            .chartYScale(domain: -1.0...1.0)
            .chartXAxis(.hidden)
            .chartYAxis(.hidden)
            .chartPlotStyle { plotArea in
                plotArea.background(Color(hex: "#0a0e1a"))
            }
            .frame(height: 40)
            .overlay(
                Rectangle()
                    .stroke(Color(hex: color).opacity(0.2), lineWidth: 1)
            )
        }
    }
}

// MARK: - Event Traffic Timeline (LLM calls, intuition, speech/thought markers)
struct EventTrafficTimeline: View {
    let noodlingId: String
    let data: [TimelinePoint]
    @ObservedObject var viewModel: NoodleScopeViewModel
    @State private var hoveredEventIndex: Int? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("EVENT TRAFFIC")
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundColor(.gray)

            // Event markers on timeline
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background
                    Rectangle()
                        .fill(Color(hex: "#0a0e1a"))
                        .border(Color(hex: "#2a3f5f"), width: 1)

                    // Event markers
                    ForEach(Array(data.enumerated()), id: \.element.id) { index, point in
                        let x = (point.timestamp / max(viewModel.maxTime, 0.1)) * geometry.size.width

                        // Event node
                        Circle()
                            .fill(eventColor(for: point))
                            .frame(width: 8, height: 8)
                            .position(x: x, y: geometry.size.height / 2)
                            .onTapGesture {
                                viewModel.selectEvent(point)
                            }
                            .onHover { hovering in
                                hoveredEventIndex = hovering ? index : nil
                            }
                            .overlay(
                                // Tooltip on hover
                                Group {
                                    if hoveredEventIndex == index {
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(point.eventType.uppercased())
                                                .font(.system(size: 9, weight: .bold, design: .monospaced))
                                            if let utterance = point.utterance {
                                                Text(String(utterance.prefix(60)) + (utterance.count > 60 ? "..." : ""))
                                                    .font(.system(size: 9, design: .monospaced))
                                                    .lineLimit(2)
                                            }
                                        }
                                        .padding(6)
                                        .background(Color(hex: "#1e2938"))
                                        .cornerRadius(4)
                                        .shadow(radius: 4)
                                        .offset(x: 0, y: -40)
                                    }
                                }
                            )
                    }

                    // Playhead line
                    let playheadX = (viewModel.playheadTime / max(viewModel.maxTime, 0.1)) * geometry.size.width
                    Rectangle()
                        .fill(.white.opacity(0.6))
                        .frame(width: 2)
                        .position(x: playheadX, y: geometry.size.height / 2)
                }
            }
            .frame(height: 30)
        }
    }

    func eventColor(for point: TimelinePoint) -> Color {
        if point.didSpeak {
            return Color(hex: "#64b5f6")  // Speech = blue
        } else if point.isThought {
            return Color.gray.opacity(0.5)  // Thought = gray
        } else if point.eventType == "enter" || point.eventType == "exit" {
            return Color(hex: "#ba68c8")  // Movement = purple
        } else if !point.facsCodes.isEmpty {
            return Color(hex: "#66bb6a")  // Expression = green
        } else {
            return Color.gray.opacity(0.3)  // Other = dim gray
        }
    }
}

// MARK: - Inspector Console Panel (bottom panel for clicked events)
struct InspectorConsolePanel: View {
    @ObservedObject var viewModel: NoodleScopeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("EVENT INSPECTOR")
                .font(.system(size: 12, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: "#64b5f6"))

            if let event = viewModel.selectedEvent {
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        // Event metadata
                        HStack {
                            Text("TYPE:")
                                .foregroundColor(.gray)
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                            Text(event.eventType.uppercased())
                                .foregroundColor(Color(hex: "#ffa726"))
                                .font(.system(size: 10, weight: .bold, design: .monospaced))

                            Spacer()

                            Text("TIME:")
                                .foregroundColor(.gray)
                                .font(.system(size: 10, design: .monospaced))
                            Text(String(format: "%.2fs", event.timestamp))
                                .foregroundColor(Color(hex: "#64b5f6"))
                                .font(.system(size: 10, weight: .bold, design: .monospaced))

                            if !event.respondingTo.isEmpty {
                                Text("|")
                                    .foregroundColor(.gray)
                                Text("RESPONDING TO:")
                                    .foregroundColor(.gray)
                                    .font(.system(size: 10, design: .monospaced))
                                Text(event.respondingTo.replacingOccurrences(of: "agent_", with: "").replacingOccurrences(of: "user_", with: ""))
                                    .foregroundColor(Color(hex: "#ba68c8"))
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                            }
                        }

                        Divider().background(.gray)

                        // FACS + Body Language
                        if !event.facsCodes.isEmpty || !event.bodyCodes.isEmpty {
                            HStack(alignment: .top, spacing: 12) {
                                if !event.facsCodes.isEmpty {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("FACIAL (FACS):")
                                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                                            .foregroundColor(.gray)
                                        ForEach(event.facsCodes.prefix(6), id: \.0) { code in
                                            Text("\(code.0): \(code.1)")
                                                .font(.system(size: 10, design: .monospaced))
                                                .foregroundColor(Color(hex: "#66bb6a"))
                                        }
                                    }
                                }

                                if !event.bodyCodes.isEmpty {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("BODY (LABAN):")
                                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                                            .foregroundColor(.gray)
                                        ForEach(event.bodyCodes.prefix(6), id: \.0) { code in
                                            Text("\(code.0): \(code.1)")
                                                .font(.system(size: 10, design: .monospaced))
                                                .foregroundColor(Color(hex: "#ffa726"))
                                        }
                                    }
                                }
                            }

                            if !event.expressionDescription.isEmpty {
                                Text(event.expressionDescription)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundColor(Color(hex: "#e0e0e0"))
                                    .italic()
                            }

                            Divider().background(.gray)
                        }

                        // Speech/Thought
                        if let utterance = event.utterance, !utterance.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(event.didSpeak ? "SPEECH:" : "PRIVATE THOUGHT:")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundColor(event.didSpeak ? Color(hex: "#64b5f6") : .gray)

                                Text(utterance)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundColor(Color(hex: "#e0e0e0"))
                            }

                            Divider().background(.gray)
                        }

                        // 5-D Affect values
                        HStack(spacing: 10) {
                            CompactMetric(label: "VALENCE", value: event.valence, color: "#66bb6a")
                            CompactMetric(label: "AROUSAL", value: event.arousal, color: "#ffa726")
                            CompactMetric(label: "FEAR", value: event.fear, color: "#ef5350")
                            CompactMetric(label: "SORROW", value: event.sorrow, color: "#9c27b0")
                            CompactMetric(label: "BOREDOM", value: event.boredom, color: "#999")
                        }

                        HStack(spacing: 10) {
                            CompactMetric(label: "SURPRISE", value: event.surprise, color: "#64b5f6")
                        }

                        // Trigger context
                        if !event.triggerText.isEmpty {
                            Divider().background(.gray)

                            VStack(alignment: .leading, spacing: 4) {
                                Text("TRIGGER CONTEXT:")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundColor(.gray)
                                Text(event.triggerText)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(Color(hex: "#999"))
                            }
                        }
                    }
                    .padding(12)
                }
            } else {
                Text("Click an event marker to inspect")
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.gray)
                    .italic()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding(12)
    }
}

// MARK: - Compact Metric (simple label + value)
struct CompactMetric: View {
    let label: String
    let value: Double
    let color: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(size: 8, design: .monospaced))
                .foregroundColor(.gray)
            Text(String(format: "%.3f", value))
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: color))
        }
        .padding(6)
        .background(Color(hex: "#0f1419"))
        .cornerRadius(4)
    }
}

// MARK: - Timeline View (Graph + Slider)
struct TimelineView: View {
    @ObservedObject var viewModel: NoodleScopeViewModel

    var body: some View {
        VStack(spacing: 0) {
            // KRUGERRAND MULTI-TRACK TIMELINE!
            if !viewModel.timelineData.isEmpty {
                ScrollView {
                    VStack(spacing: 4) {
                        // Track 1: Valence
                        AffectTrack(title: "VALENCE", color: "#66bb6a",
                                   data: viewModel.timelineData,
                                   valueKeyPath: \.valence,
                                   playheadTime: viewModel.playheadTime,
                                   maxTime: viewModel.maxTime)

                        // Track 2: Arousal
                        AffectTrack(title: "AROUSAL", color: "#ffa726",
                                   data: viewModel.timelineData,
                                   valueKeyPath: \.arousal,
                                   playheadTime: viewModel.playheadTime,
                                   maxTime: viewModel.maxTime)

                        // Track 3: Fear
                        AffectTrack(title: "FEAR", color: "#ef5350",
                                   data: viewModel.timelineData,
                                   valueKeyPath: \.fear,
                                   playheadTime: viewModel.playheadTime,
                                   maxTime: viewModel.maxTime)

                        // Track 4: Sorrow
                        AffectTrack(title: "SORROW", color: "#9c27b0",
                                   data: viewModel.timelineData,
                                   valueKeyPath: \.sorrow,
                                   playheadTime: viewModel.playheadTime,
                                   maxTime: viewModel.maxTime)

                        // Track 5: Surprise (with spike highlights!)
                        AffectTrack(title: "SURPRISE", color: "#64b5f6",
                                   data: viewModel.timelineData,
                                   valueKeyPath: \.surprise,
                                   playheadTime: viewModel.playheadTime,
                                   maxTime: viewModel.maxTime,
                                   highlightSpikes: true)
                    }
                    .padding(.horizontal, 12)
                }
                .frame(height: 500)
                .background(Color(hex: "#0a0e1a"))

                // Event Inspector at playhead position
                EventInspectorPanel(viewModel: viewModel)
                    .frame(height: 200)
                    .background(Color(hex: "#131824"))
            } else {
                Text("Loading session data...")
                    .foregroundColor(Color(hex: "#64b5f6"))
                    .frame(height: 350)
            }

            // Playhead slider - PERFECTLY ALIGNED!
            HStack(spacing: 12) {
                Text(String(format: "%.1fs / %.1fs",
                           viewModel.playheadTime,
                           viewModel.maxTime))
                    .font(.system(size: 14, design: .monospaced))
                    .foregroundColor(Color(hex: "#64b5f6"))
                    .frame(width: 100)

                Slider(value: $viewModel.playheadTime,
                       in: 0...viewModel.maxTime,
                       step: 0.1)
                    .tint(Color(hex: "#64b5f6"))

                Button("Ask @Kimmie") {
                    Task { await viewModel.askKimmie() }
                }
                .buttonStyle(.borderedProminent)
                .tint(Color(hex: "#ba68c8"))
            }
            .padding(12)
            .background(Color(hex: "#131824"))
            .border(Color(hex: "#2a3f5f"), width: 1)
        }
    }
}

// MARK: - Affect Track (Single labeled timeline track)
struct AffectTrack: View {
    let title: String
    let color: String
    let data: [TimelinePoint]
    let valueKeyPath: KeyPath<TimelinePoint, Double>
    let playheadTime: Double
    let maxTime: Double
    var highlightSpikes: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            // Label
            Text(title)
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: color))
                .padding(.leading, 4)

            // Mini chart for this dimension
            Chart {
                ForEach(data) { point in
                    LineMark(
                        x: .value("Time", point.timestamp),
                        y: .value(title, point[keyPath: valueKeyPath])
                    )
                    .foregroundStyle(Color(hex: color))
                    .lineStyle(StrokeStyle(lineWidth: 2))

                    // Highlight spikes for surprise track
                    if highlightSpikes && point[keyPath: valueKeyPath] > 0.3 {
                        PointMark(
                            x: .value("Time", point.timestamp),
                            y: .value(title, point[keyPath: valueKeyPath])
                        )
                        .foregroundStyle(Color(hex: "#ffeb3b"))
                        .symbol(.circle)
                        .symbolSize(60)
                    }
                }

                // Playhead
                RuleMark(x: .value("Playhead", playheadTime))
                    .foregroundStyle(.white.opacity(0.6))
                    .lineStyle(StrokeStyle(lineWidth: 1))
            }
            .chartXScale(domain: 0...max(maxTime, 0.1))
            .chartYScale(domain: -1.0...1.0)
            .chartXAxis(.hidden)
            .chartYAxis {
                AxisMarks(values: [-1, 0, 1]) { value in
                    AxisValueLabel()
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(Color.gray.opacity(0.6))
                }
            }
            .chartPlotStyle { plotArea in
                plotArea.background(Color(hex: "#0f1419"))
            }
            .frame(height: 60)
            .overlay(
                Rectangle()
                    .stroke(Color(hex: color).opacity(0.3), lineWidth: 1)
            )
        }
    }
}

// MARK: - Event Inspector (shows details at playhead)
struct EventInspectorPanel: View {
    @ObservedObject var viewModel: NoodleScopeViewModel

    var currentEvent: TimelinePoint? {
        viewModel.timelineData.min(by: { abs($0.timestamp - viewModel.playheadTime) < abs($1.timestamp - viewModel.playheadTime) })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("EVENT INSPECTOR @ \(String(format: "%.1fs", viewModel.playheadTime))")
                .font(.system(size: 12, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: "#64b5f6"))
                .padding(.bottom, 4)

            if let event = currentEvent {
                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        // Event type and trigger
                        HStack {
                            Text("Type:")
                                .foregroundColor(.gray)
                                .font(.system(size: 11, design: .monospaced))
                            Text(event.eventType.uppercased())
                                .foregroundColor(Color(hex: "#ffa726"))
                                .font(.system(size: 11, weight: .bold, design: .monospaced))

                            Spacer()

                            if !event.respondingTo.isEmpty {
                                Text("→")
                                    .foregroundColor(.gray)
                                Text(event.respondingTo.replacingOccurrences(of: "agent_", with: "").replacingOccurrences(of: "user_", with: ""))
                                    .foregroundColor(Color(hex: "#ba68c8"))
                                    .font(.system(size: 11, design: .monospaced))
                            }
                        }

                        // FACS codes
                        if !event.facsCodes.isEmpty {
                            HStack(alignment: .top, spacing: 4) {
                                Text("FACE:")
                                    .foregroundColor(.gray)
                                    .font(.system(size: 10, design: .monospaced))
                                Text(event.facsCodes.map { $0.0 }.joined(separator: ", "))
                                    .foregroundColor(Color(hex: "#66bb6a"))
                                    .font(.system(size: 10, design: .monospaced))
                            }
                        }

                        // Body codes
                        if !event.bodyCodes.isEmpty {
                            HStack(alignment: .top, spacing: 4) {
                                Text("BODY:")
                                    .foregroundColor(.gray)
                                    .font(.system(size: 10, design: .monospaced))
                                Text(event.bodyCodes.map { $0.0 }.joined(separator: ", "))
                                    .foregroundColor(Color(hex: "#ffa726"))
                                    .font(.system(size: 10, design: .monospaced))
                            }
                        }

                        // Expression description
                        if !event.expressionDescription.isEmpty {
                            Text(event.expressionDescription)
                                .foregroundColor(Color(hex: "#e0e0e0"))
                                .font(.system(size: 11, design: .monospaced))
                                .italic()
                        }

                        // Speech/Thought
                        if let utterance = event.utterance, !utterance.isEmpty {
                            HStack(alignment: .top, spacing: 4) {
                                Text(event.didSpeak ? "SPEECH:" : "THOUGHT:")
                                    .foregroundColor(event.didSpeak ? Color(hex: "#64b5f6") : .gray)
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                                Text(utterance)
                                    .foregroundColor(Color(hex: "#e0e0e0"))
                                    .font(.system(size: 11, design: .monospaced))
                            }
                        }

                        // Affect values
                        HStack(spacing: 12) {
                            MetricBadge(label: "V", value: event.valence, color: "#66bb6a")
                            MetricBadge(label: "A", value: event.arousal, color: "#ffa726")
                            MetricBadge(label: "F", value: event.fear, color: "#ef5350")
                            MetricBadge(label: "So", value: event.sorrow, color: "#9c27b0")
                            MetricBadge(label: "B", value: event.boredom, color: "#999")
                            Spacer()
                            MetricBadge(label: "SURPRISE", value: event.surprise, color: "#64b5f6", large: true)
                        }

                        // Trigger text
                        if !event.triggerText.isEmpty {
                            HStack(alignment: .top, spacing: 4) {
                                Text("TRIGGER:")
                                    .foregroundColor(.gray)
                                    .font(.system(size: 10, design: .monospaced))
                                Text(event.triggerText)
                                    .foregroundColor(Color(hex: "#999"))
                                    .font(.system(size: 10, design: .monospaced))
                            }
                        }
                    }
                    .padding(8)
                }
            } else {
                Text("No event at playhead")
                    .foregroundColor(.gray)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding(12)
    }
}

// MARK: - Metric Badge (compact value display)
struct MetricBadge: View {
    let label: String
    let value: Double
    let color: String
    var large: Bool = false

    var body: some View {
        HStack(spacing: 2) {
            Text(label)
                .font(.system(size: large ? 11 : 9, weight: .bold, design: .monospaced))
                .foregroundColor(.gray)
            Text(String(format: "%.2f", value))
                .font(.system(size: large ? 12 : 10, weight: large ? .bold : .regular, design: .monospaced))
                .foregroundColor(Color(hex: color))
        }
        .padding(.horizontal, large ? 8 : 4)
        .padding(.vertical, large ? 4 : 2)
        .background(Color(hex: "#0f1419"))
        .cornerRadius(4)
    }
}

// MARK: - Other Panels (Simplified)
struct PhenomenalStatePanel: View {
    @ObservedObject var viewModel: NoodleScopeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Phenomenal State (40-D)")
                .font(.system(size: 14, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: "#64b5f6"))

            if let state = viewModel.currentPhenomenalState {
                ScrollView {
                    VStack(spacing: 8) {
                        StateSection(title: "Fast Layer (16-D)",
                                   values: Array(state.prefix(16)),
                                   color: "#66bb6a")
                        StateSection(title: "Medium Layer (16-D)",
                                   values: Array(state.dropFirst(16).prefix(16)),
                                   color: "#ffa726")
                        StateSection(title: "Slow Layer (8-D)",
                                   values: Array(state.dropFirst(32)),
                                   color: "#ba68c8")
                    }
                }
            } else {
                Text("No state loaded")
                    .foregroundColor(.gray)
                    .italic()
            }
        }
        .padding(16)
    }
}

struct StateSection: View {
    let title: String
    let values: [Double]
    let color: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(.gray)
                .textCase(.uppercase)

            ForEach(Array(values.enumerated()), id: \.offset) { index, value in
                HStack(spacing: 4) {
                    Text("\(index)")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.gray)
                        .frame(width: 20, alignment: .trailing)

                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Rectangle()
                                .fill(Color(hex: "#1e2938"))

                            Rectangle()
                                .fill(Color(hex: color))
                                .frame(width: geo.size.width * CGFloat((value + 1) / 2))
                        }
                    }
                    .frame(height: 16)
                    .cornerRadius(3)

                    Text(String(format: "%.2f", value))
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundColor(.black)
                        .frame(width: 40)
                }
            }
        }
    }
}

struct ConversationContextPanel: View {
    @ObservedObject var viewModel: NoodleScopeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("🗨️ LLM CONVERSATION CONTEXT")
                .font(.system(size: 12, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: "#ffa726"))

            if viewModel.conversationContext.isEmpty {
                Text("No conversation context at this timestep")
                    .foregroundColor(.gray)
                    .italic()
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(viewModel.conversationContext, id: \.role) { message in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(message.role.uppercased())
                                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                                    .foregroundColor(.gray)
                                Text(message.content)
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundColor(Color(hex: "#b0b0b0"))
                            }
                            .padding(8)
                            .background(Color(hex: "#1e2938"))
                            .cornerRadius(4)
                        }
                    }
                }
            }
        }
        .padding(12)
        .background(Color(hex: "#131824"))
    }
}

struct MetricsPanel: View {
    @ObservedObject var viewModel: NoodleScopeViewModel

    var body: some View {
        HStack(spacing: 12) {
            MetricCard(label: "HSI", value: viewModel.metrics.hsi, status: "warning_up")
            MetricCard(label: "SURPRISE", value: viewModel.metrics.surprise, status: "Low")
            MetricCard(label: "VALENCE", value: viewModel.metrics.valence, status: "Positive")
            MetricCard(label: "AROUSAL", value: viewModel.metrics.arousal, status: "Excited")
            MetricCard(label: "CHEAP THRILLS", value: viewModel.metrics.cheapThrills, status: "Philosophical")
        }
        .padding(16)
        .background(Color(hex: "#1a2230"))
    }
}

struct MetricCard: View {
    let label: String
    let value: Double
    let status: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(.gray)
                .textCase(.uppercase)

            Text(String(format: "%.4f", value))
                .font(.system(size: 20, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: "#64b5f6"))

            Text(status)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color(hex: "#131824"))
        .cornerRadius(6)
    }
}

struct OperationsPanel: View {
    @ObservedObject var viewModel: NoodleScopeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("OPERATIONS TIMELINE")
                .font(.system(size: 12, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: "#64b5f6"))

            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(viewModel.operations, id: \.timestamp) { op in
                        Text("[\(op.timestamp)] \(op.agent) | \(op.operation) | \(op.duration)ms")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(colorForDuration(op.duration))
                    }
                }
            }
        }
        .padding(12)
        .background(Color(hex: "#0a0e1a"))
    }

    func colorForDuration(_ duration: Double) -> Color {
        if duration < 100 { return Color(hex: "#66bb6a") }
        if duration < 500 { return Color(hex: "#ffa726") }
        return Color(hex: "#ef5350")
    }
}

// MARK: - View Model
@MainActor
class NoodleScopeViewModel: ObservableObject {
    @Published var sessions: [String] = []
    @Published var selectedSession: String = ""
    @Published var agents: [String] = []
    @Published var selectedAgent: String = ""
    @Published var timelineData: [TimelinePoint] = []
    @Published var playheadTime: Double = 0
    @Published var maxTime: Double = 18.1
    @Published var currentPhenomenalState: [Double]?
    @Published var conversationContext: [ConversationMessage] = []
    @Published var metrics = Metrics()
    @Published var operations: [Operation] = []
    @Published var selectedEvent: TimelinePoint? = nil  // For inspector console
    @Published var enlightenedAgents: Set<String> = []  // Track enlightenment status

    private let baseURL = "http://localhost:8081/api"

    func loadSessions() async {
        // Fetch from API
        guard let url = URL(string: "\(baseURL)/profiler/live-session") else { return }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let response = try JSONDecoder().decode(SessionResponse.self, from: data)

            self.sessions = ["🔴 LIVE SESSION"]
            self.selectedSession = "🔴 LIVE SESSION"
            self.agents = Array(response.timelines.keys).sorted()

            if !agents.isEmpty {
                self.selectedAgent = agents[0]
                await loadTimeline()
            }
        } catch {
            print("Error loading sessions: \(error)")
        }
    }

    func loadTimeline() async {
        guard !selectedAgent.isEmpty else { return }
        guard let url = URL(string: "\(baseURL)/profiler/live-session") else { return }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            let response = try JSONDecoder().decode(SessionResponse.self, from: data)

            if let timeline = response.timelines[selectedAgent] {
                self.timelineData = timeline.map { point in
                    // Parse FACS codes
                    let facs = (point.facs_codes ?? []).map { arr -> (String, String) in
                        (arr.count > 0 ? arr[0] : "", arr.count > 1 ? arr[1] : "")
                    }

                    // Parse body codes
                    let body = (point.body_codes ?? []).map { arr -> (String, String) in
                        (arr.count > 0 ? arr[0] : "", arr.count > 1 ? arr[1] : "")
                    }

                    // Convert conversation context (placeholder for now)
                    let context: [ConversationMessage] = []

                    return TimelinePoint(
                        timestamp: point.timestamp,
                        valence: point.affect.valence,
                        arousal: point.affect.arousal,
                        fear: point.affect.fear,
                        sorrow: point.affect.sorrow ?? 0.0,
                        boredom: point.affect.boredom ?? 0.0,
                        surprise: point.surprise,
                        facsCodes: facs,
                        bodyCodes: body,
                        expressionDescription: point.expression_description ?? "",
                        utterance: point.utterance,
                        didSpeak: point.did_speak ?? false,
                        isThought: !(point.did_speak ?? false),
                        eventType: point.event_type ?? "unknown",
                        respondingTo: point.responding_to ?? "",
                        conversationContext: context,
                        triggerText: point.event ?? ""
                    )
                }

                if let last = timeline.last {
                    self.maxTime = last.timestamp
                }
            }
        } catch {
            print("Error loading timeline: \(error)")
        }
    }

    func askKimmie() async {
        // TODO: Implement Kimmie interpretation
        print("Asking Kimmie about timeline segment...")
    }

    func selectEvent(_ event: TimelinePoint) {
        selectedEvent = event
        playheadTime = event.timestamp  // Jump playhead to clicked event
    }

    func isEnlightened(_ agentId: String) -> Bool {
        return enlightenedAgents.contains(agentId)
    }
}

// MARK: - Data Models
struct TimelinePoint: Identifiable {
    let id = UUID()
    let timestamp: Double
    let valence: Double
    let arousal: Double
    let fear: Double
    let sorrow: Double
    let boredom: Double
    let surprise: Double

    // FACS/Body Language (Krugerrand upgrade!)
    let facsCodes: [(String, String)]  // [(code, description)]
    let bodyCodes: [(String, String)]
    let expressionDescription: String

    // Speech/Thought/Action
    let utterance: String?
    let didSpeak: Bool
    let isThought: Bool
    let eventType: String
    let respondingTo: String

    // Full context for inspector
    let conversationContext: [ConversationMessage]
    let triggerText: String
}

struct ConversationMessage {
    let role: String
    let content: String
}

struct Metrics {
    var hsi: Double = 0.0
    var surprise: Double = 0.0
    var valence: Double = 0.68
    var arousal: Double = 0.54
    var cheapThrills: Double = 0.0
}

struct Operation {
    let timestamp: String
    let agent: String
    let operation: String
    let duration: Double
}

struct SessionResponse: Codable {
    let timelines: [String: [TimelineData]]
}

struct TimelineData: Codable {
    let timestamp: Double
    let affect: Affect
    let surprise: Double
    let did_speak: Bool?
    let utterance: String?
    let event: String?
    let event_type: String?
    let responding_to: String?
    let facs_codes: [[String]]?  // Array of [code, description] pairs
    let body_codes: [[String]]?
    let expression_description: String?
}

struct Affect: Codable {
    let valence: Double
    let arousal: Double
    let fear: Double
    let sorrow: Double?
    let boredom: Double?
}

// MARK: - Color Extension
extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)
        }

        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue:  Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

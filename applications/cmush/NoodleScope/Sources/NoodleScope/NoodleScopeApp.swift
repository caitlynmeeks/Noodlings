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
            // Header
            HeaderView(viewModel: viewModel)
                .frame(height: 60)
                .background(Color(hex: "#131824"))

            HStack(spacing: 0) {
                // Left panel: Phenomenal State
                PhenomenalStatePanel(viewModel: viewModel)
                    .frame(width: 300)
                    .background(Color(hex: "#131824"))

                // Main timeline area
                VStack(spacing: 0) {
                    // Conversation context
                    ConversationContextPanel(viewModel: viewModel)
                        .frame(maxHeight: 200)

                    // Metrics
                    MetricsPanel(viewModel: viewModel)
                        .frame(height: 120)

                    // Timeline graph + slider
                    TimelineView(viewModel: viewModel)

                    // Operations console
                    OperationsPanel(viewModel: viewModel)
                        .frame(height: 250)
                }
            }
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

            // Agent tabs
            HStack(spacing: 4) {
                ForEach(viewModel.agents, id: \.self) { agent in
                    Button(action: {
                        viewModel.selectedAgent = agent
                    }) {
                        Text(agent.replacingOccurrences(of: "agent_", with: ""))
                            .font(.system(size: 14, design: .monospaced))
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(viewModel.selectedAgent == agent ?
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

// MARK: - Timeline View (Graph + Slider)
struct TimelineView: View {
    @ObservedObject var viewModel: NoodleScopeViewModel

    var body: some View {
        VStack(spacing: 0) {
            // Timeline chart
            if !viewModel.timelineData.isEmpty {
                Chart {
                    ForEach(viewModel.timelineData) { point in
                        LineMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Valence", point.valence)
                        )
                        .foregroundStyle(Color(hex: "#66bb6a"))

                        LineMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Arousal", point.arousal)
                        )
                        .foregroundStyle(Color(hex: "#ffa726"))

                        LineMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Fear", point.fear)
                        )
                        .foregroundStyle(Color(hex: "#ef5350"))

                        LineMark(
                            x: .value("Time", point.timestamp),
                            y: .value("Surprise", point.surprise)
                        )
                        .foregroundStyle(Color(hex: "#64b5f6"))
                    }

                    // Playhead line
                    RuleMark(x: .value("Playhead", viewModel.playheadTime))
                        .foregroundStyle(.white.opacity(0.8))
                        .lineStyle(StrokeStyle(lineWidth: 2, dash: [5, 5]))
                }
                .chartXAxis {
                    AxisMarks(values: .automatic) { value in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 1))
                            .foregroundStyle(Color(hex: "#2a3f5f"))
                        AxisValueLabel()
                            .foregroundStyle(Color(hex: "#e0e0e0"))
                    }
                }
                .chartYAxis {
                    AxisMarks(values: .automatic) { value in
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 1))
                            .foregroundStyle(Color(hex: "#2a3f5f"))
                        AxisValueLabel()
                            .foregroundStyle(Color(hex: "#e0e0e0"))
                    }
                }
                .chartPlotStyle { plotArea in
                    plotArea
                        .background(Color(hex: "#0a0e1a"))
                }
                .frame(height: 350)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
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
            Text("⚡ OPERATIONS TIMELINE")
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
                    TimelinePoint(
                        timestamp: point.timestamp,
                        valence: point.affect.valence,
                        arousal: point.affect.arousal,
                        fear: point.affect.fear,
                        surprise: point.surprise
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
}

// MARK: - Data Models
struct TimelinePoint: Identifiable {
    let id = UUID()
    let timestamp: Double
    let valence: Double
    let arousal: Double
    let fear: Double
    let surprise: Double
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
}

struct Affect: Codable {
    let valence: Double
    let arousal: Double
    let fear: Double
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

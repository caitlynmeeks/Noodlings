import SwiftUI
import WebKit

// App delegate to keep app in foreground
class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

@main
struct NoodleStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var workspace = WorkspaceManager()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(workspace)
        }
        .defaultSize(width: 1400, height: 900)
        .commands {
            CommandMenu("Window") {
                Button("New noodleMUSH View") {
                    workspace.openPanel(.noodleMUSH)
                }
                Button("New NoodleScope") {
                    workspace.openPanel(.noodleScope)
                }
                Button("New Drama Manager") {
                    workspace.openPanel(.dramaManager)
                }
                Button("New Console") {
                    workspace.openPanel(.console)
                }
                Button("New Inspector") {
                    workspace.openPanel(.inspector)
                }
                Divider()
                Button("Reset Layout") {
                    workspace.resetLayout()
                }
            }
        }
    }
}

// MARK: - Main Content View
struct ContentView: View {
    @EnvironmentObject var workspace: WorkspaceManager
    @State private var selection: PanelType = .noodleMUSH

    var body: some View {
        NavigationSplitView {
            // Sidebar with panel types
            List(PanelType.allCases, id: \.self, selection: $selection) { panelType in
                Label(panelType.title, systemImage: panelType.icon)
                    .tag(panelType)
            }
            .navigationTitle("NoodleStudio")
        } detail: {
            // Main content area - show selected panel
            PanelContainerView(panelType: selection)
        }
        .frame(minWidth: 1200, minHeight: 800)
    }
}

// MARK: - Panel Container
struct PanelContainerView: View {
    let panelType: PanelType

    var body: some View {
        VStack(spacing: 0) {
            // Panel toolbar
            HStack {
                Text(panelType.title)
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
                Button(action: {}) {
                    Image(systemName: "sidebar.right")
                }
                .buttonStyle(.borderless)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))

            Divider()

            // Panel content
            panelContent
        }
    }

    @ViewBuilder
    private var panelContent: some View {
        switch panelType {
        case .noodleMUSH:
            NoodleMUSHPanel()
        case .noodleScope:
            NoodleScopePanel()
        case .dramaManager:
            DramaManagerPanel()
        case .console:
            ConsolePanel()
        case .inspector:
            InspectorPanel()
        }
    }
}

// MARK: - noodleMUSH Panel (WebView)
struct NoodleMUSHPanel: View {
    @State private var webView = WKWebView()
    @State private var isLoading = true

    var body: some View {
        ZStack {
            WebView(webView: $webView, isLoading: $isLoading)
                .onAppear {
                    if let url = URL(string: "http://localhost:8080") {
                        webView.load(URLRequest(url: url))
                    }
                }

            if isLoading {
                VStack {
                    ProgressView()
                        .scaleEffect(1.5)
                    Text("Loading noodleMUSH...")
                        .padding()
                        .foregroundColor(.secondary)
                }
            }
        }
    }
}

struct WebView: NSViewRepresentable {
    @Binding var webView: WKWebView
    @Binding var isLoading: Bool

    func makeNSView(context: Context) -> WKWebView {
        webView.navigationDelegate = context.coordinator
        webView.allowsMagnification = true

        // Enable developer extras
        if let key = "developerExtrasEnabled" as CFString as? String {
            webView.configuration.preferences.setValue(true, forKey: key)
        }

        // Enable text input
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            webView.window?.makeFirstResponder(webView)
        }

        return webView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {
        // Make sure webview can receive input
        nsView.window?.makeFirstResponder(nsView)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, WKNavigationDelegate {
        var parent: WebView

        init(_ parent: WebView) {
            self.parent = parent
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            parent.isLoading = false
        }

        func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) {
            parent.isLoading = true
        }
    }
}

// MARK: - NoodleScope Panel
struct NoodleScopePanel: View {
    var body: some View {
        VStack {
            Text("NoodleScope")
                .font(.title)
            Text("Timeline visualization will go here")
                .foregroundColor(.secondary)
            Text("(We can integrate the existing NoodleScope code)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Drama Manager Panel
struct DramaManagerPanel: View {
    @State private var plays: [String] = []
    @State private var selectedPlay: String?

    var body: some View {
        HSplitView {
            // Play list
            List(plays, id: \.self, selection: $selectedPlay) { play in
                Text(play)
            }
            .frame(minWidth: 200)

            // Play editor
            VStack {
                if let selected = selectedPlay {
                    Text("Editing: \(selected)")
                        .font(.title2)
                    Spacer()
                    Text("Play editor UI goes here")
                        .foregroundColor(.secondary)
                    Spacer()
                } else {
                    Text("Select a play to edit")
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .onAppear {
            loadPlays()
        }
    }

    func loadPlays() {
        // TODO: Load from ../cmush/plays/
        plays = ["the_pugs_play", "the_noodle_watch", "the_mask_of_many_faces"]
    }
}

// MARK: - Console Panel
struct ConsolePanel: View {
    @State private var logLines: [LogLine] = []
    @State private var filterLevel: LogLevel = .all

    var body: some View {
        VStack(spacing: 0) {
            // Filter toolbar
            HStack {
                Text("Filter:")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Picker("", selection: $filterLevel) {
                    ForEach(LogLevel.allCases, id: \.self) { level in
                        Text(level.title).tag(level)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 300)

                Spacer()

                Button("Clear") {
                    logLines.removeAll()
                }
            }
            .padding()

            Divider()

            // Log output
            ScrollView {
                ScrollViewReader { proxy in
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(filteredLogs) { line in
                            LogLineView(line: line)
                                .id(line.id)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                }
            }
            .background(Color(NSColor.textBackgroundColor))
            .font(.system(.body, design: .monospaced))
        }
        .onAppear {
            startLogTail()
        }
    }

    var filteredLogs: [LogLine] {
        if filterLevel == .all {
            return logLines
        }
        return logLines.filter { $0.level == filterLevel }
    }

    func startLogTail() {
        // TODO: Tail the log file from cmush/logs/
        // For now, add sample logs
        logLines = [
            LogLine(level: .info, message: "NoodleStudio started", timestamp: Date()),
            LogLine(level: .info, message: "Connected to noodleMUSH server", timestamp: Date()),
            LogLine(level: .debug, message: "Agent agent_callie loaded", timestamp: Date())
        ]
    }
}

struct LogLineView: View {
    let line: LogLine

    var body: some View {
        HStack(spacing: 8) {
            Text(line.timestamp, style: .time)
                .foregroundColor(.secondary)
                .frame(width: 80, alignment: .leading)

            Image(systemName: line.level.icon)
                .foregroundColor(line.level.color)
                .frame(width: 20)

            Text(line.message)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Inspector Panel
struct InspectorPanel: View {
    @State private var agents: [String] = []
    @State private var selectedAgent: String?

    var body: some View {
        HSplitView {
            // Agent list
            List(agents, id: \.self, selection: $selectedAgent) { agent in
                Text(agent.replacingOccurrences(of: "agent_", with: ""))
            }
            .frame(minWidth: 150)

            // Agent inspector
            if let agent = selectedAgent {
                AgentInspectorView(agentId: agent)
            } else {
                VStack {
                    Image(systemName: "person.crop.circle")
                        .font(.system(size: 60))
                        .foregroundColor(.secondary)
                    Text("Select an agent to inspect")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear {
            loadAgents()
        }
    }

    func loadAgents() {
        // TODO: Fetch from API
        agents = ["agent_callie", "agent_toad", "agent_phi"]
    }
}

struct AgentInspectorView: View {
    let agentId: String
    @State private var valence: Double = 0.68
    @State private var arousal: Double = 0.54
    @State private var fear: Double = 0.1

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Header
                VStack(alignment: .leading, spacing: 4) {
                    Text(agentId.replacingOccurrences(of: "agent_", with: "").capitalized)
                        .font(.title)
                    Text("Agent Parameters")
                        .foregroundColor(.secondary)
                }

                Divider()

                // Affect parameters
                GroupBox("Affect Vector") {
                    VStack(alignment: .leading, spacing: 12) {
                        SliderRow(label: "Valence", value: $valence, range: -1...1)
                        SliderRow(label: "Arousal", value: $arousal, range: 0...1)
                        SliderRow(label: "Fear", value: $fear, range: 0...1)
                    }
                    .padding()
                }

                // Personality
                GroupBox("Personality") {
                    VStack(alignment: .leading, spacing: 12) {
                        SliderRow(label: "Extraversion", value: .constant(0.7), range: 0...1)
                        SliderRow(label: "Curiosity", value: .constant(0.65), range: 0...1)
                        SliderRow(label: "Social Orientation", value: .constant(0.8), range: 0...1)
                    }
                    .padding()
                }

                // Actions
                HStack {
                    Button("Apply Changes") {
                        // TODO: Send to API
                    }
                    .buttonStyle(.borderedProminent)

                    Button("Reset") {
                        // TODO: Reset values
                    }
                }

                Spacer()
            }
            .padding()
        }
    }
}

struct SliderRow: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.caption)
                Spacer()
                Text(String(format: "%.2f", value))
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .monospacedDigit()
            }
            Slider(value: $value, in: range)
        }
    }
}

// MARK: - Data Models
enum PanelType: String, CaseIterable {
    case noodleMUSH = "noodlemush"
    case noodleScope = "scope"
    case dramaManager = "drama"
    case console = "console"
    case inspector = "inspector"

    var title: String {
        switch self {
        case .noodleMUSH: return "noodleMUSH"
        case .noodleScope: return "NoodleScope"
        case .dramaManager: return "Drama Manager"
        case .console: return "Console"
        case .inspector: return "Inspector"
        }
    }

    var icon: String {
        switch self {
        case .noodleMUSH: return "network"
        case .noodleScope: return "waveform.path.ecg"
        case .dramaManager: return "theatermasks"
        case .console: return "terminal"
        case .inspector: return "slider.horizontal.3"
        }
    }
}

class WorkspaceManager: ObservableObject {
    @Published var openPanels: Set<PanelType> = [.noodleMUSH]

    func openPanel(_ panel: PanelType) {
        openPanels.insert(panel)
    }

    func closePanel(_ panel: PanelType) {
        openPanels.remove(panel)
    }

    func resetLayout() {
        openPanels = [.noodleMUSH, .console]
    }
}

struct LogLine: Identifiable {
    let id = UUID()
    let level: LogLevel
    let message: String
    let timestamp: Date
}

enum LogLevel: String, CaseIterable {
    case all = "all"
    case debug = "debug"
    case info = "info"
    case warning = "warning"
    case error = "error"

    var title: String {
        switch self {
        case .all: return "All"
        case .debug: return "Debug"
        case .info: return "Info"
        case .warning: return "Warning"
        case .error: return "Error"
        }
    }

    var icon: String {
        switch self {
        case .all: return "circle"
        case .debug: return "ladybug"
        case .info: return "info.circle"
        case .warning: return "exclamationmark.triangle"
        case .error: return "xmark.circle"
        }
    }

    var color: Color {
        switch self {
        case .all: return .secondary
        case .debug: return .purple
        case .info: return .blue
        case .warning: return .orange
        case .error: return .red
        }
    }
}

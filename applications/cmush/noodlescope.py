"""
NoodleScope - Real-time Noodling Consciousness Visualization

A consciousness oscilloscope for monitoring phenomenal states across multiple agents.
Watch the Noodles noodle in real-time!

Features:
- Multi-agent, multi-channel visualization
- Real-time streaming of 40-D phenomenal states
- Surprise spike detection
- Identity salience tracking
- Timeline scrubbing and playback
- Event log

Author: Sir Claude, Knight of the Noodlings
Date: November 2025
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from collections import deque
from datetime import datetime
import json
import asyncio
import websockets
from threading import Thread
import time

# Configuration
MAX_POINTS = 300  # Show last 5 minutes at 1Hz
UPDATE_INTERVAL = 500  # ms

class NoodleScope:
    """Real-time consciousness visualization dashboard."""

    def __init__(self, host='localhost', port=8765):
        """
        Initialize NoodleScope.

        Args:
            host: WebSocket server host
            port: WebSocket server port
        """
        self.host = host
        self.port = port

        # Data buffers for each agent
        self.agents = {}  # {agent_id: AgentBuffer}

        # Recording
        self.recording = []
        self.recording_enabled = True

        # Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        self.setup_api_endpoints()

    def setup_layout(self):
        """Build the dashboard layout."""
        self.app.layout = html.Div([
            html.Div([
                html.H1("🧠 NoodleScope", style={'display': 'inline-block', 'margin-right': '20px'}),
                html.Span("Noodlings Multi-Agent Consciousness Oscilloscope",
                         style={'color': '#666', 'font-size': '14px'})
            ], style={'padding': '20px', 'background': '#f0f0f0', 'border-bottom': '2px solid #00ff00'}),

            # Controls
            html.Div([
                html.Button('▶ Play', id='play-btn', n_clicks=0,
                           style={'margin': '5px', 'padding': '10px 20px'}),
                html.Button('⏸ Pause', id='pause-btn', n_clicks=0,
                           style={'margin': '5px', 'padding': '10px 20px'}),
                html.Button('💾 Save Recording', id='save-btn', n_clicks=0,
                           style={'margin': '5px', 'padding': '10px 20px'}),
                html.Span(id='status', children='● Live',
                         style={'margin-left': '20px', 'color': '#00ff00', 'font-weight': 'bold'})
            ], style={'padding': '10px 20px', 'background': '#f9f9f9'}),

            # Agent selector
            html.Div([
                html.Label('Agents:', style={'font-weight': 'bold', 'margin-right': '10px'}),
                dcc.Checklist(
                    id='agent-selector',
                    options=[],
                    value=[],
                    inline=True,
                    style={'display': 'inline-block'}
                )
            ], style={'padding': '10px 20px'}),

            # Main visualization
            dcc.Graph(id='phenomenal-state-graph',
                     style={'height': '600px'}),

            # Event log
            html.Div([
                html.H3("Event Log"),
                html.Div(id='event-log',
                        style={
                            'height': '200px',
                            'overflow-y': 'scroll',
                            'border': '1px solid #ddd',
                            'padding': '10px',
                            'background': '#000',
                            'color': '#0f0',
                            'font-family': 'monospace',
                            'font-size': '12px'
                        })
            ], style={'padding': '20px'}),

            # Hidden interval component for updates
            dcc.Interval(
                id='interval-component',
                interval=UPDATE_INTERVAL,
                n_intervals=0
            ),

            # Hidden store for data
            dcc.Store(id='scope-data', data={'agents': {}, 'events': []})
        ])

    def setup_callbacks(self):
        """Setup Dash callbacks for interactivity."""

        @self.app.callback(
            [Output('phenomenal-state-graph', 'figure'),
             Output('event-log', 'children'),
             Output('agent-selector', 'options'),
             Output('agent-selector', 'value')],
            [Input('interval-component', 'n_intervals'),
             Input('agent-selector', 'value')],
            [State('agent-selector', 'options')]
        )
        def update_graphs(n, selected_agents, current_options):
            """Update all visualizations."""

            # Build agent options
            agent_options = [
                {'label': f"{aid.replace('agent_', '').title()} ({aid})", 'value': aid}
                for aid in self.agents.keys()
            ]

            # Auto-select new agents
            if not selected_agents or len(agent_options) > len(current_options):
                selected_agents = [opt['value'] for opt in agent_options]

            # Create figure with subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Fast Layer (Arousal)', 'Surprise Spikes',
                               'Slow Layer (Personality)', 'Identity Salience'),
                vertical_spacing=0.08,
                row_heights=[0.3, 0.25, 0.25, 0.2]
            )

            # Plot each selected agent
            colors = ['#00ff00', '#ff00ff', '#00ffff', '#ffaa00', '#ff0000']

            for i, agent_id in enumerate(selected_agents):
                if agent_id not in self.agents:
                    continue

                agent_data = self.agents[agent_id]
                color = colors[i % len(colors)]
                agent_name = agent_id.replace('agent_', '').title()

                # Get data
                timestamps = list(agent_data.timestamps)
                fast_arousal = [state[1] if len(state) > 1 else 0 for state in agent_data.fast_states]
                surprises = list(agent_data.surprises)
                slow_curiosity = [state[2] if len(state) > 2 else 0 for state in agent_data.slow_states]
                identity_saliences = list(agent_data.identity_saliences)

                # Fast layer arousal
                fig.add_trace(
                    go.Scatter(x=timestamps, y=fast_arousal,
                              mode='lines', name=f'{agent_name} arousal',
                              line=dict(color=color, width=2)),
                    row=1, col=1
                )

                # Surprise spikes
                fig.add_trace(
                    go.Scatter(x=timestamps, y=surprises,
                              mode='lines', name=f'{agent_name} surprise',
                              line=dict(color=color, width=2),
                              fill='tozeroy'),
                    row=2, col=1
                )

                # Slow layer (personality drift)
                fig.add_trace(
                    go.Scatter(x=timestamps, y=slow_curiosity,
                              mode='lines', name=f'{agent_name} curiosity',
                              line=dict(color=color, width=2)),
                    row=3, col=1
                )

                # Identity salience
                fig.add_trace(
                    go.Scatter(x=timestamps, y=identity_saliences,
                              mode='lines', name=f'{agent_name} salience',
                              line=dict(color=color, width=2),
                              fill='tozeroy'),
                    row=4, col=1
                )

            # Update layout
            fig.update_xaxes(title_text="Time", row=4, col=1)
            fig.update_yaxes(title_text="Value", range=[0, 1])

            fig.update_layout(
                height=600,
                showlegend=True,
                hovermode='x unified',
                template='plotly_dark',
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#2d2d2d'
            )

            # Build event log
            event_items = []
            for event in list(self.recording)[-20:]:  # Last 20 events
                timestamp = event.get('timestamp', '')
                agent = event.get('agent', 'unknown').replace('agent_', '').title()
                text = event.get('text', '')
                event_type = event.get('type', 'event')

                # Color code by type
                color = '#00ff00'
                if event_type == 'surprise_spike':
                    color = '#ffff00'
                elif event_type == 'name_mentioned':
                    color = '#00ffff'
                elif event_type == 'high_salience':
                    color = '#ff00ff'

                event_items.append(
                    html.Div(
                        f"[{timestamp}] {agent}: {text}",
                        style={'color': color, 'margin-bottom': '5px'}
                    )
                )

            return fig, event_items, agent_options, selected_agents

    def add_agent(self, agent_id):
        """Register a new agent for tracking."""
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentBuffer()

    def update_agent_state(self, agent_id, phenomenal_state, surprise, identity_salience=0.0):
        """
        Update agent's phenomenal state.

        Args:
            agent_id: Agent identifier
            phenomenal_state: Full 40-D state (fast 16 + medium 16 + slow 8)
            surprise: Current surprise value
            identity_salience: Current identity salience
        """
        self.add_agent(agent_id)

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Parse 40-D state
        fast_state = phenomenal_state[:16] if len(phenomenal_state) >= 16 else []
        medium_state = phenomenal_state[16:32] if len(phenomenal_state) >= 32 else []
        slow_state = phenomenal_state[32:40] if len(phenomenal_state) >= 40 else []

        # Store in buffer
        agent_buffer = self.agents[agent_id]
        agent_buffer.timestamps.append(timestamp)
        agent_buffer.fast_states.append(fast_state)
        agent_buffer.medium_states.append(medium_state)
        agent_buffer.slow_states.append(slow_state)
        agent_buffer.surprises.append(surprise)
        agent_buffer.identity_saliences.append(identity_salience)

    def log_event(self, agent_id, event_type, text):
        """
        Log an event to the timeline.

        Args:
            agent_id: Agent identifier
            event_type: Event type (surprise_spike, name_mentioned, etc.)
            text: Event description
        """
        if self.recording_enabled:
            self.recording.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'agent': agent_id,
                'type': event_type,
                'text': text
            })

    def setup_api_endpoints(self):
        """Setup Flask API endpoints for receiving live data."""
        from flask import request, jsonify

        @self.app.server.route('/api/update_state', methods=['POST'])
        def update_state_endpoint():
            """Receive phenomenal state updates from agents."""
            try:
                data = request.get_json()
                agent_id = data.get('agent_id')
                phenomenal_state = data.get('phenomenal_state', [])
                surprise = data.get('surprise', 0.0)
                identity_salience = data.get('identity_salience', 0.0)

                self.update_agent_state(agent_id, phenomenal_state, surprise, identity_salience)

                return jsonify({'status': 'ok'}), 200
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @self.app.server.route('/api/log_event', methods=['POST'])
        def log_event_endpoint():
            """Receive event logs from agents."""
            try:
                data = request.get_json()
                agent_id = data.get('agent_id')
                event_type = data.get('event_type', 'event')
                text = data.get('text', '')

                self.log_event(agent_id, event_type, text)

                return jsonify({'status': 'ok'}), 200
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500

    def run(self, debug=False, port=8050):
        """
        Start the NoodleScope dashboard.

        Args:
            debug: Enable debug mode
            port: Dashboard port
        """
        print(f"🧠 NoodleScope starting on http://localhost:{port}")
        self.app.run(debug=debug, port=port, host='0.0.0.0')


class AgentBuffer:
    """Circular buffer for agent state history."""

    def __init__(self, maxlen=MAX_POINTS):
        self.timestamps = deque(maxlen=maxlen)
        self.fast_states = deque(maxlen=maxlen)
        self.medium_states = deque(maxlen=maxlen)
        self.slow_states = deque(maxlen=maxlen)
        self.surprises = deque(maxlen=maxlen)
        self.identity_saliences = deque(maxlen=maxlen)


# Example usage / test mode
if __name__ == '__main__':
    scope = NoodleScope()

    # Add some test data
    scope.add_agent('agent_callie')
    scope.add_agent('agent_phi')

    # Simulate data
    for i in range(100):
        t = i * 0.1

        # Callie: smooth oscillation
        callie_state = [0.5 + 0.3 * np.sin(t)] * 16 + [0.5] * 16 + [0.5 + 0.2 * np.sin(t * 0.1)] * 8
        callie_surprise = 0.3 + 0.2 * np.sin(t * 2)
        scope.update_agent_state('agent_callie', callie_state, callie_surprise, 0.5)

        # Phi: erratic kitten behavior
        phi_state = [0.7 + 0.3 * np.random.random()] * 16 + [0.6] * 16 + [0.8] * 8
        phi_surprise = 0.2 + 0.6 * np.random.random()
        scope.update_agent_state('agent_phi', phi_state, phi_surprise, 0.7 if i % 10 == 0 else 0.3)

        if i % 10 == 0:
            scope.log_event('agent_phi', 'high_salience', '*meows loudly*')

    # Run dashboard
    scope.run(debug=True)

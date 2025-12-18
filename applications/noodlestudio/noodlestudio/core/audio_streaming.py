"""
Audio Streaming - WebSocket handler for real-time audio I/O.

Handles:
- Microphone audio streaming from browser/client
- Real-time transcription feedback
- TTS audio streaming to client
- Voice activity detection integration

WebSocket Protocol:
    Client -> Server:
        {type: "audio_chunk", data: base64, sample_rate: 16000, timestamp: float}
        {type: "start_listening"}
        {type: "stop_listening"}
        {type: "speak", text: "Hello!"}
        {type: "interrupt"}
        {type: "configure", voice: "...", sensitivity: 0.5}

    Server -> Client:
        {type: "transcription", text: "...", is_final: bool, confidence: float}
        {type: "audio_out", data: base64, sample_rate: 22050}
        {type: "state", listening: bool, speaking: bool}
        {type: "error", message: "..."}

Integration:
    - Add to existing cMUSH server via add_audio_handlers()
    - Or run standalone via AudioStreamingServer

Author: Commander Spock + Cadet Caity
Date: December 17, 2025
"""

import asyncio
import base64
import json
import logging
import time
from typing import Dict, Any, Optional, Set, Callable
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class AudioConnection:
    """Track state for an audio WebSocket connection."""
    websocket: Any
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    is_listening: bool = False
    is_speaking: bool = False
    connected_at: float = 0.0
    last_audio_at: float = 0.0
    chunks_received: int = 0
    bytes_received: int = 0


class AudioStreamingHandler:
    """
    WebSocket handler for audio streaming.

    Integrates with AudioStreamFacet for processing.
    Can be added to existing WebSocket server or run standalone.
    """

    def __init__(self):
        """Initialize audio streaming handler."""
        self.connections: Dict[str, AudioConnection] = {}
        self._audio_facet = None
        self._transcription_client = None
        self._tts_client = None

        # Callback for routing transcriptions to agents
        self._on_transcription: Optional[Callable] = None

        # Audio buffer for accumulating chunks before transcription
        self._audio_buffers: Dict[str, bytearray] = {}
        self._buffer_duration_ms: Dict[str, float] = {}

        # Configuration
        self.chunk_duration_ms = 250
        self.min_transcription_duration_ms = 500
        self.max_buffer_duration_ms = 30000  # 30 seconds max

    # ========== Setup ==========

    def set_audio_facet(self, facet):
        """Connect to AudioStreamFacet for processing."""
        self._audio_facet = facet
        logger.info("[AudioStreaming] Connected to AudioStreamFacet")

    def set_transcription_client(self, client):
        """Set transcription client (Whisper)."""
        self._transcription_client = client
        logger.info("[AudioStreaming] Transcription client set")

    def set_tts_client(self, client):
        """Set TTS client (ElevenLabs, etc.)."""
        self._tts_client = client
        logger.info("[AudioStreaming] TTS client set")

    def on_transcription(self, callback: Callable[[str, str, str], None]):
        """
        Set callback for transcription results.

        Args:
            callback: Function(user_id, agent_id, text) -> None
        """
        self._on_transcription = callback

    # ========== WebSocket Handlers ==========

    async def handle_audio_message(self, websocket, message: Dict[str, Any],
                                   user_id: str, agent_id: Optional[str] = None) -> Optional[Dict]:
        """
        Handle audio-related WebSocket message.

        Args:
            websocket: WebSocket connection
            message: Parsed message dict
            user_id: User ID
            agent_id: Target agent ID (if any)

        Returns:
            Response dict or None
        """
        msg_type = message.get('type')
        conn_id = str(id(websocket))

        # Ensure connection tracking
        if conn_id not in self.connections:
            self.connections[conn_id] = AudioConnection(
                websocket=websocket,
                user_id=user_id,
                agent_id=agent_id,
                connected_at=time.time()
            )
            self._audio_buffers[conn_id] = bytearray()
            self._buffer_duration_ms[conn_id] = 0.0

        conn = self.connections[conn_id]

        try:
            if msg_type == 'audio_chunk':
                return await self._handle_audio_chunk(conn_id, conn, message)

            elif msg_type == 'start_listening':
                return await self._handle_start_listening(conn_id, conn)

            elif msg_type == 'stop_listening':
                return await self._handle_stop_listening(conn_id, conn)

            elif msg_type == 'speak':
                text = message.get('text', '')
                return await self._handle_speak(conn_id, conn, text)

            elif msg_type == 'interrupt':
                return await self._handle_interrupt(conn_id, conn)

            elif msg_type == 'configure':
                return await self._handle_configure(conn_id, conn, message)

            else:
                return None  # Not an audio message

        except Exception as e:
            logger.error(f"[AudioStreaming] Error handling {msg_type}: {e}")
            return {'type': 'error', 'message': str(e)}

    async def _handle_audio_chunk(self, conn_id: str, conn: AudioConnection,
                                  message: Dict) -> Optional[Dict]:
        """Handle incoming audio chunk."""
        if not conn.is_listening:
            return {'type': 'error', 'message': 'Not listening'}

        # Decode base64 audio
        audio_b64 = message.get('data', '')
        sample_rate = message.get('sample_rate', 16000)
        timestamp = message.get('timestamp', time.time())

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            logger.error(f"[AudioStreaming] Failed to decode audio: {e}")
            return {'type': 'error', 'message': 'Invalid audio data'}

        # Update stats
        conn.last_audio_at = timestamp
        conn.chunks_received += 1
        conn.bytes_received += len(audio_bytes)

        # Calculate chunk duration (16-bit PCM mono)
        duration_ms = (len(audio_bytes) / 2 / sample_rate) * 1000

        # Accumulate in buffer
        self._audio_buffers[conn_id].extend(audio_bytes)
        self._buffer_duration_ms[conn_id] += duration_ms

        # Push to audio facet if available
        if self._audio_facet:
            await self._audio_facet.push_audio_chunk(
                audio_bytes, sample_rate, duration_ms
            )

        # Check if we should transcribe
        buffer_duration = self._buffer_duration_ms[conn_id]

        # Transcribe if buffer is large enough or max duration reached
        if buffer_duration >= self.max_buffer_duration_ms:
            return await self._transcribe_buffer(conn_id, conn, sample_rate)

        return None  # No immediate response, transcription happens on stop or timeout

    async def _handle_start_listening(self, conn_id: str,
                                      conn: AudioConnection) -> Dict:
        """Start listening for audio input."""
        conn.is_listening = True

        # Clear buffer
        self._audio_buffers[conn_id] = bytearray()
        self._buffer_duration_ms[conn_id] = 0.0

        # Start audio facet listening
        if self._audio_facet:
            await self._audio_facet.listen()

        logger.info(f"[AudioStreaming] Started listening for {conn.user_id}")

        return {
            'type': 'state',
            'listening': True,
            'speaking': conn.is_speaking
        }

    async def _handle_stop_listening(self, conn_id: str,
                                     conn: AudioConnection) -> Dict:
        """Stop listening and transcribe accumulated audio."""
        conn.is_listening = False

        # Transcribe buffer if there's content
        result = None
        if self._buffer_duration_ms.get(conn_id, 0) >= self.min_transcription_duration_ms:
            result = await self._transcribe_buffer(conn_id, conn)

        # Stop audio facet listening
        if self._audio_facet:
            await self._audio_facet.stop_listening()

        logger.info(f"[AudioStreaming] Stopped listening for {conn.user_id}")

        if result:
            return result

        return {
            'type': 'state',
            'listening': False,
            'speaking': conn.is_speaking
        }

    async def _handle_speak(self, conn_id: str, conn: AudioConnection,
                           text: str) -> Dict:
        """Synthesize and stream speech."""
        if not text:
            return {'type': 'error', 'message': 'No text provided'}

        if not self._tts_client:
            return {'type': 'error', 'message': 'TTS not configured'}

        conn.is_speaking = True

        try:
            # Synthesize audio
            audio_bytes = await self._tts_client.synthesize(text)

            if audio_bytes:
                # Stream audio back to client
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                await conn.websocket.send(json.dumps({
                    'type': 'audio_out',
                    'data': audio_b64,
                    'sample_rate': self._tts_client.sample_rate,
                    'text': text
                }))

                logger.info(f"[AudioStreaming] Sent TTS audio for: {text[:30]}...")

        except Exception as e:
            logger.error(f"[AudioStreaming] TTS error: {e}")
            return {'type': 'error', 'message': f'TTS failed: {e}'}

        finally:
            conn.is_speaking = False

        return {
            'type': 'state',
            'listening': conn.is_listening,
            'speaking': False
        }

    async def _handle_interrupt(self, conn_id: str, conn: AudioConnection) -> Dict:
        """Interrupt current TTS playback."""
        conn.is_speaking = False

        if self._audio_facet:
            await self._audio_facet.interrupt()

        logger.info(f"[AudioStreaming] Interrupted for {conn.user_id}")

        return {
            'type': 'state',
            'listening': conn.is_listening,
            'speaking': False,
            'interrupted': True
        }

    async def _handle_configure(self, conn_id: str, conn: AudioConnection,
                               message: Dict) -> Dict:
        """Handle configuration updates."""
        if 'voice' in message and self._tts_client:
            self._tts_client.set_voice(message['voice'])

        if 'sensitivity' in message and self._audio_facet:
            self._audio_facet.set_sensitivity(message['sensitivity'])

        logger.info(f"[AudioStreaming] Configuration updated for {conn.user_id}")

        return {'type': 'configured', 'success': True}

    async def _transcribe_buffer(self, conn_id: str, conn: AudioConnection,
                                 sample_rate: int = 16000) -> Optional[Dict]:
        """Transcribe accumulated audio buffer."""
        buffer = self._audio_buffers.get(conn_id)
        if not buffer:
            return None

        # Clear buffer
        audio_bytes = bytes(buffer)
        self._audio_buffers[conn_id] = bytearray()
        self._buffer_duration_ms[conn_id] = 0.0

        if not self._transcription_client:
            logger.warning("[AudioStreaming] No transcription client configured")
            return {'type': 'error', 'message': 'Transcription not configured'}

        try:
            # Transcribe
            result = await self._transcription_client.transcribe(audio_bytes, sample_rate)

            text = result.get('text', '').strip()
            confidence = result.get('confidence', 0.0)
            language = result.get('language', 'en')

            if text:
                logger.info(f"[AudioStreaming] Transcribed: {text}")

                # Notify callback
                if self._on_transcription and conn.user_id:
                    self._on_transcription(conn.user_id, conn.agent_id, text)

                return {
                    'type': 'transcription',
                    'text': text,
                    'is_final': True,
                    'confidence': confidence,
                    'language': language
                }

        except Exception as e:
            logger.error(f"[AudioStreaming] Transcription error: {e}")
            return {'type': 'error', 'message': f'Transcription failed: {e}'}

        return None

    # ========== Connection Management ==========

    def on_disconnect(self, websocket):
        """Handle WebSocket disconnection."""
        conn_id = str(id(websocket))

        if conn_id in self.connections:
            conn = self.connections[conn_id]
            logger.info(f"[AudioStreaming] Disconnected: {conn.user_id} "
                       f"(chunks={conn.chunks_received}, bytes={conn.bytes_received})")
            del self.connections[conn_id]

        if conn_id in self._audio_buffers:
            del self._audio_buffers[conn_id]
        if conn_id in self._buffer_duration_ms:
            del self._buffer_duration_ms[conn_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        total_chunks = sum(c.chunks_received for c in self.connections.values())
        total_bytes = sum(c.bytes_received for c in self.connections.values())

        return {
            'active_connections': len(self.connections),
            'total_chunks_received': total_chunks,
            'total_bytes_received': total_bytes,
            'listening_count': sum(1 for c in self.connections.values() if c.is_listening),
            'speaking_count': sum(1 for c in self.connections.values() if c.is_speaking)
        }


# ========== Integration Helpers ==========

def add_audio_handlers(server, handler: AudioStreamingHandler):
    """
    Add audio streaming handlers to existing cMUSH server.

    Usage in server.py:
        from noodlestudio.core.audio_streaming import AudioStreamingHandler, add_audio_handlers

        audio_handler = AudioStreamingHandler()
        add_audio_handlers(self, audio_handler)

    Then in handle_connection:
        if msg_type in ['audio_chunk', 'start_listening', 'stop_listening', 'speak', 'interrupt', 'configure']:
            response = await audio_handler.handle_audio_message(websocket, data, user_id, agent_id)
            if response:
                await websocket.send(json.dumps(response))
    """
    server.audio_handler = handler

    # Set up transcription routing to agent input
    def route_transcription(user_id: str, agent_id: str, text: str):
        """Route transcription as agent input."""
        if hasattr(server, 'handle_say'):
            # Queue as if user said it
            asyncio.create_task(
                server.route_to_agent(user_id, agent_id, text)
            )

    handler.on_transcription(route_transcription)

    logger.info("[AudioStreaming] Handlers added to server")


# ========== Standalone Server ==========

class AudioStreamingServer:
    """
    Standalone WebSocket server for audio streaming.

    Use when not integrating with existing server.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        """
        Initialize standalone audio server.

        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.handler = AudioStreamingHandler()

    def set_transcription_client(self, client):
        """Set transcription client."""
        self.handler.set_transcription_client(client)

    def set_tts_client(self, client):
        """Set TTS client."""
        self.handler.set_tts_client(client)

    async def handle_connection(self, websocket, path=None):
        """Handle WebSocket connection."""
        user_id = f"user_{id(websocket)}"
        logger.info(f"[AudioServer] New connection: {user_id}")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.handler.handle_audio_message(
                        websocket, data, user_id
                    )
                    if response:
                        await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON'
                    }))

        except Exception as e:
            logger.error(f"[AudioServer] Connection error: {e}")

        finally:
            self.handler.on_disconnect(websocket)

    async def start(self):
        """Start the audio streaming server."""
        import websockets

        logger.info(f"[AudioServer] Starting on ws://{self.host}:{self.port}")

        async with websockets.serve(self.handle_connection, self.host, self.port):
            logger.info("[AudioServer] Ready for connections")
            await asyncio.Future()  # Run forever


# ========== Browser Client Helper ==========

BROWSER_CLIENT_JS = '''
/**
 * Audio streaming client for browser.
 *
 * Usage:
 *   const client = new AudioStreamClient('ws://localhost:8766');
 *   await client.connect();
 *   await client.startListening();
 *   client.onTranscription = (text) => console.log('You said:', text);
 */
class AudioStreamClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.mediaStream = null;
        this.audioContext = null;
        this.processor = null;
        this.isListening = false;

        // Callbacks
        this.onTranscription = null;
        this.onAudioOut = null;
        this.onStateChange = null;
        this.onError = null;
    }

    async connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url);
            this.ws.onopen = () => resolve();
            this.ws.onerror = (e) => reject(e);
            this.ws.onmessage = (e) => this._handleMessage(JSON.parse(e.data));
        });
    }

    async startListening() {
        // Get microphone access
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        // Set up audio processing
        this.audioContext = new AudioContext({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);

        // Create processor for chunking
        await this.audioContext.audioWorklet.addModule('audio-processor.js');
        this.processor = new AudioWorkletNode(this.audioContext, 'audio-processor');

        this.processor.port.onmessage = (e) => {
            if (this.isListening && this.ws.readyState === WebSocket.OPEN) {
                const audioData = e.data;
                const b64 = btoa(String.fromCharCode(...new Uint8Array(audioData.buffer)));
                this.ws.send(JSON.stringify({
                    type: 'audio_chunk',
                    data: b64,
                    sample_rate: 16000,
                    timestamp: Date.now() / 1000
                }));
            }
        };

        source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);

        // Tell server we're listening
        this.ws.send(JSON.stringify({ type: 'start_listening' }));
        this.isListening = true;
    }

    stopListening() {
        this.isListening = false;
        this.ws.send(JSON.stringify({ type: 'stop_listening' }));

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(t => t.stop());
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }

    speak(text) {
        this.ws.send(JSON.stringify({ type: 'speak', text }));
    }

    interrupt() {
        this.ws.send(JSON.stringify({ type: 'interrupt' }));
    }

    _handleMessage(msg) {
        switch (msg.type) {
            case 'transcription':
                if (this.onTranscription) this.onTranscription(msg.text, msg.is_final);
                break;
            case 'audio_out':
                if (this.onAudioOut) this.onAudioOut(msg.data, msg.sample_rate);
                this._playAudio(msg.data, msg.sample_rate);
                break;
            case 'state':
                if (this.onStateChange) this.onStateChange(msg);
                break;
            case 'error':
                if (this.onError) this.onError(msg.message);
                break;
        }
    }

    async _playAudio(b64Data, sampleRate) {
        const audioData = Uint8Array.from(atob(b64Data), c => c.charCodeAt(0));
        const audioContext = new AudioContext({ sampleRate });
        const audioBuffer = await audioContext.decodeAudioData(audioData.buffer);
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start();
    }
}
'''


def get_browser_client_js() -> str:
    """Get JavaScript client code for browser integration."""
    return BROWSER_CLIENT_JS

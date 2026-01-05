"""
WebView UI Component - Embedded web browser

Displays web content within the UI canvas. Useful for:
- Documentation panels
- External dashboards
- Interactive web content
- HTML-based UI elements

Author: Caitlyn + Claude
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from ..component import UIComponent, register_component


@register_component
@dataclass
class WebView(UIComponent):
    """
    Embedded web browser component.

    Properties:
        url: The URL to load (http://, https://, or file://)
        html: Raw HTML content to display (alternative to url)
        zoom_factor: Zoom level (1.0 = 100%)
        scrollbars: Whether to show scrollbars
        javascript_enabled: Whether to enable JavaScript
        background: Background color (shown before content loads)

    Events:
        onLoad: Fired when page finishes loading
        onError: Fired on load error
        onUrlChanged: Fired when URL changes (navigation)

    Example YAML:
        - type: WebView
          name: docPanel
          url: "https://docs.example.com"
          geometry: {x: 0, y: 0, width: 400, height: 300}
          zoom_factor: 1.0
          events:
            onLoad:
              action: call_script
              script: |
                console.log("Page loaded:", event.url);
    """

    component_type: str = field(default="WebView", init=False)

    # Content source (either url OR html, not both)
    url: str = ""
    html: str = ""

    # Display options
    zoom_factor: float = 1.0
    scrollbars: bool = True
    javascript_enabled: bool = True
    background: str = "#1e1e1e"

    def __post_init__(self):
        """Set default geometry for WebView."""
        super().__post_init__()
        if self.geometry.width == 100:  # Default width
            self.geometry.width = 400
        if self.geometry.height == 32:  # Default height
            self.geometry.height = 300

    def load_url(self, url: str):
        """Load a new URL."""
        self.url = url
        self.html = ""  # Clear html when loading URL

    def load_html(self, html: str, base_url: str = ""):
        """Load raw HTML content."""
        self.html = html
        self.url = base_url  # Base URL for relative links

    def reload(self):
        """Reload the current content."""
        # This is handled by the renderer
        pass

    def go_back(self):
        """Navigate back in history."""
        # This is handled by the renderer
        pass

    def go_forward(self):
        """Navigate forward in history."""
        # This is handled by the renderer
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = super().to_dict()

        # Add WebView-specific properties
        if self.url:
            data['url'] = self.url
        if self.html:
            data['html'] = self.html
        if self.zoom_factor != 1.0:
            data['zoom_factor'] = self.zoom_factor
        if not self.scrollbars:
            data['scrollbars'] = self.scrollbars
        if not self.javascript_enabled:
            data['javascript_enabled'] = self.javascript_enabled
        if self.background != "#1e1e1e":
            data['background'] = self.background

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebView':
        """Deserialize from dictionary."""
        webview = cls(
            name=data.get('name', 'webview'),
            url=data.get('url', ''),
            html=data.get('html', ''),
            zoom_factor=data.get('zoom_factor', 1.0),
            scrollbars=data.get('scrollbars', True),
            javascript_enabled=data.get('javascript_enabled', True),
            background=data.get('background', '#1e1e1e'),
        )

        # Load base class properties
        webview._load_base_properties(data)

        return webview

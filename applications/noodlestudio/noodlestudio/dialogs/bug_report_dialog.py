"""
Bug Report Dialog for NoodleStudio
==================================

Collects bug reports and crash information for submission to GitHub Issues.
"""

import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QComboBox, QLineEdit, QCheckBox, QWidget,
    QGroupBox, QFormLayout, QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

from .. import __version__


# Bug report submission endpoint
BUG_REPORT_ENDPOINT = "https://noodlings-auth.noodlings.workers.dev/api/bug-report"


class SystemInfoCollector:
    """Collects system information for bug reports."""

    @staticmethod
    def collect() -> Dict[str, Any]:
        """Gather system information."""
        info = {
            "noodlestudio_version": __version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "timestamp": datetime.now().isoformat(),
        }

        # Try to get Qt version
        try:
            from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
            info["qt_version"] = QT_VERSION_STR
            info["pyqt_version"] = PYQT_VERSION_STR
        except:
            pass

        # Try to get GPU info on macOS
        if platform.system() == "Darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType", "-json"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_data = json.loads(result.stdout)
                    displays = gpu_data.get("SPDisplaysDataType", [])
                    if displays:
                        info["gpu"] = displays[0].get("sppci_model", "Unknown")
            except:
                pass

        return info

    @staticmethod
    def format_for_display(info: Dict[str, Any]) -> str:
        """Format system info for display in the dialog."""
        lines = [
            f"NoodleStudio: {info.get('noodlestudio_version', 'Unknown')}",
            f"Platform: {info.get('platform', 'Unknown')}",
            f"Python: {info.get('python_version', 'Unknown').split()[0]}",
        ]
        if "qt_version" in info:
            lines.append(f"Qt: {info['qt_version']}")
        if "gpu" in info:
            lines.append(f"GPU: {info['gpu']}")
        return "\n".join(lines)


class BugReportDialog(QDialog):
    """
    Dialog for reporting bugs to GitHub Issues via Cloudflare Worker proxy.
    """

    report_submitted = pyqtSignal(str)  # Emits issue URL on success

    SEVERITY_LEVELS = [
        ("Crash", "crash", "Application crashes or data loss"),
        ("Major", "major", "Feature broken, no workaround"),
        ("Minor", "minor", "Feature broken, workaround exists"),
        ("Cosmetic", "cosmetic", "Visual or text issue"),
    ]

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        crash_info: Optional[Dict[str, Any]] = None
    ):
        super().__init__(parent)
        self.crash_info = crash_info
        self.system_info = SystemInfoCollector.collect()
        self.network_manager = QNetworkAccessManager(self)

        self.setWindowTitle("Report a Bug" if not crash_info else "Crash Report")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setMinimumHeight(450)

        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
            }
            QLabel {
                color: #cccccc;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 6px;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border-color: #76AF6A;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 8px;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
            QGroupBox {
                color: #888888;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QCheckBox {
                color: #cccccc;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)

        self._setup_ui()
        self._connect_signals()

        # Pre-fill if crash report
        if crash_info:
            self._prefill_crash_info()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_text = "Report a Bug" if not self.crash_info else "NoodleStudio Encountered an Error"
        title = QLabel(title_text)
        title.setFont(QFont("", 16, QFont.Weight.DemiBold))
        title.setStyleSheet("color: #ffffff;")
        layout.addWidget(title)

        if self.crash_info:
            subtitle = QLabel("Would you like to send a crash report to help us fix this issue?")
            subtitle.setStyleSheet("color: #888888; font-size: 12px;")
            subtitle.setWordWrap(True)
            layout.addWidget(subtitle)

        # Summary field
        layout.addWidget(QLabel("Summary:"))
        self.summary_edit = QLineEdit()
        self.summary_edit.setPlaceholderText("Brief description of the issue")
        layout.addWidget(self.summary_edit)

        # Severity
        severity_layout = QHBoxLayout()
        severity_layout.addWidget(QLabel("Severity:"))
        self.severity_combo = QComboBox()
        for name, _, tooltip in self.SEVERITY_LEVELS:
            self.severity_combo.addItem(name)
        self.severity_combo.setToolTip("How severe is this issue?")
        if self.crash_info:
            self.severity_combo.setCurrentIndex(0)  # Crash
        severity_layout.addWidget(self.severity_combo)
        severity_layout.addStretch()
        layout.addLayout(severity_layout)

        # Description
        layout.addWidget(QLabel("Steps to Reproduce:"))
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText(
            "1. What were you doing?\n"
            "2. What did you expect to happen?\n"
            "3. What actually happened?"
        )
        self.description_edit.setMinimumHeight(100)
        layout.addWidget(self.description_edit)

        # Stack trace (for crashes)
        if self.crash_info:
            layout.addWidget(QLabel("Error Details:"))
            self.stacktrace_edit = QTextEdit()
            self.stacktrace_edit.setReadOnly(True)
            self.stacktrace_edit.setMaximumHeight(100)
            self.stacktrace_edit.setStyleSheet(
                self.stacktrace_edit.styleSheet() +
                "font-family: monospace; font-size: 11px;"
            )
            layout.addWidget(self.stacktrace_edit)

        # System info group
        info_group = QGroupBox("System Information (will be included)")
        info_layout = QVBoxLayout(info_group)
        self.system_info_label = QLabel(SystemInfoCollector.format_for_display(self.system_info))
        self.system_info_label.setStyleSheet("font-family: monospace; font-size: 11px; color: #888888;")
        info_layout.addWidget(self.system_info_label)
        layout.addWidget(info_group)

        # Include logs checkbox
        self.include_logs_check = QCheckBox("Include recent console logs")
        self.include_logs_check.setChecked(True)
        layout.addWidget(self.include_logs_check)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumWidth(80)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d3d3d;
                color: #cccccc;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)
        button_layout.addWidget(self.cancel_btn)

        self.submit_btn = QPushButton("Submit Report")
        self.submit_btn.setMinimumWidth(120)
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #76AF6A;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #86BF7A;
            }
            QPushButton:disabled {
                background-color: #4a4a4a;
                color: #888888;
            }
        """)
        button_layout.addWidget(self.submit_btn)

        layout.addLayout(button_layout)

    def _connect_signals(self):
        self.cancel_btn.clicked.connect(self.reject)
        self.submit_btn.clicked.connect(self._submit_report)
        self.summary_edit.textChanged.connect(self._validate_form)

    def _validate_form(self):
        """Enable submit button only if summary is provided."""
        has_summary = bool(self.summary_edit.text().strip())
        self.submit_btn.setEnabled(has_summary)

    def _prefill_crash_info(self):
        """Pre-fill dialog with crash information."""
        if not self.crash_info:
            return

        # Set summary from exception
        exc_type = self.crash_info.get("exception_type", "Unknown Error")
        exc_msg = self.crash_info.get("exception_message", "")
        self.summary_edit.setText(f"Crash: {exc_type}: {exc_msg[:50]}")

        # Set stack trace
        if hasattr(self, 'stacktrace_edit'):
            self.stacktrace_edit.setPlainText(
                self.crash_info.get("traceback", "No traceback available")
            )

    def _collect_logs(self) -> str:
        """Collect recent console logs if available."""
        try:
            # Try to get logs from the main window's console panel
            main_window = self.parent()
            if main_window and hasattr(main_window, 'console_panel'):
                console = main_window.console_panel
                if hasattr(console, 'get_recent_logs'):
                    return console.get_recent_logs(max_lines=50)
        except:
            pass
        return ""

    def _build_report(self) -> Dict[str, Any]:
        """Build the bug report payload."""
        severity_idx = self.severity_combo.currentIndex()
        severity_label = self.SEVERITY_LEVELS[severity_idx][1]

        report = {
            "summary": self.summary_edit.text().strip(),
            "description": self.description_edit.toPlainText().strip(),
            "severity": severity_label,
            "system_info": self.system_info,
        }

        if self.crash_info:
            report["crash_info"] = self.crash_info

        if self.include_logs_check.isChecked():
            logs = self._collect_logs()
            if logs:
                report["console_logs"] = logs

        return report

    def _submit_report(self):
        """Submit the bug report to the backend."""
        self.submit_btn.setEnabled(False)
        self.submit_btn.setText("Submitting...")

        report = self._build_report()

        # Create request
        request = QNetworkRequest()
        request.setUrl(Qt.QUrl(BUG_REPORT_ENDPOINT))
        request.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader,
            "application/json"
        )

        # Send POST request
        reply = self.network_manager.post(
            request,
            json.dumps(report).encode('utf-8')
        )
        reply.finished.connect(lambda: self._handle_response(reply))

    def _handle_response(self, reply: QNetworkReply):
        """Handle the submission response."""
        self.submit_btn.setText("Submit Report")
        self.submit_btn.setEnabled(True)

        if reply.error() == QNetworkReply.NetworkError.NoError:
            try:
                response = json.loads(reply.readAll().data().decode('utf-8'))
                issue_url = response.get("issue_url", "")

                QMessageBox.information(
                    self,
                    "Report Submitted",
                    f"Thank you! Your bug report has been submitted.\n\n"
                    f"Track it at:\n{issue_url}" if issue_url else
                    "Thank you! Your bug report has been submitted."
                )

                self.report_submitted.emit(issue_url)
                self.accept()

            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Submission Error",
                    f"Report submitted but couldn't parse response: {e}"
                )
                self.accept()
        else:
            error_msg = reply.errorString()

            # Check if it's a connection error (endpoint not set up yet)
            if "Connection refused" in error_msg or "Host not found" in error_msg:
                # Fallback: copy to clipboard
                self._fallback_to_clipboard()
            else:
                QMessageBox.warning(
                    self,
                    "Submission Failed",
                    f"Could not submit report: {error_msg}\n\n"
                    "Please try again later or report manually on GitHub."
                )

        reply.deleteLater()

    def _fallback_to_clipboard(self):
        """If submission fails, offer to copy report to clipboard."""
        report = self._build_report()

        # Format as GitHub issue markdown
        md = f"## {report['summary']}\n\n"
        md += f"**Severity:** {report['severity']}\n\n"

        if report.get('description'):
            md += f"### Steps to Reproduce\n{report['description']}\n\n"

        if report.get('crash_info'):
            md += f"### Error Details\n```\n{report['crash_info'].get('traceback', 'N/A')}\n```\n\n"

        md += "### System Information\n```\n"
        for key, value in report['system_info'].items():
            md += f"{key}: {value}\n"
        md += "```\n"

        clipboard = QApplication.clipboard()
        clipboard.setText(md)

        QMessageBox.information(
            self,
            "Report Copied",
            "Bug report backend is not available.\n\n"
            "The report has been copied to your clipboard.\n"
            "Please paste it into a new GitHub issue at:\n"
            "https://github.com/noodlings-ai/noodlings/issues/new"
        )
        self.accept()


def show_bug_report_dialog(parent: Optional[QWidget] = None) -> Optional[str]:
    """
    Show the bug report dialog and return the issue URL if submitted.
    """
    dialog = BugReportDialog(parent)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.report_submitted
    return None


def show_crash_report_dialog(
    parent: Optional[QWidget],
    exc_type: type,
    exc_value: Exception,
    exc_tb
) -> bool:
    """
    Show a crash report dialog for an unhandled exception.
    Returns True if user submitted the report.
    """
    crash_info = {
        "exception_type": exc_type.__name__ if exc_type else "Unknown",
        "exception_message": str(exc_value) if exc_value else "",
        "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
    }

    dialog = BugReportDialog(parent, crash_info=crash_info)
    return dialog.exec() == QDialog.DialogCode.Accepted

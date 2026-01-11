# ──────────────────────────────────────────────────────────────
#
#   UI Test Runner - Automated UI testing using computer use
#
#   Uses NoodleCode's computer use infrastructure to run
#   automated tests that actually click the UI.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.testing
# PURPOSE:  UI test automation
# LAYER:    Studio / Testing
# ──────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from .ui_test_runner import UITestRunner, TestResult, TestPhaseResult
from .ui_test_actions import UITestActions
from .ui_test_targets import UITestTargetResolver
from .ui_test_assertions import UITestAssertions

__all__ = [
    'UITestRunner',
    'TestResult',
    'TestPhaseResult',
    'UITestActions',
    'UITestTargetResolver',
    'UITestAssertions',
]

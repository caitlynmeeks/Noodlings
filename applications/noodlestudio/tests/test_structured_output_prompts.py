# ------------------------------------------------------------------
#   Structured Output Prompt Tests
#
#   Verifies that all three noodling assemblies contain the
#   OUTPUT FORMAT instruction block in their Response facet prompts,
#   and that Juanita's assembly has been standardized to the same
#   topology as Ajo and Krampus (no Perception facet).
#
# ------------------------------------------------------------------
# MODULE:   applications.noodlestudio.tests.test_structured_output_prompts
# PURPOSE:  Commit 1 -- OUTPUT FORMAT blocks + Juanita standardization
# LAYER:    Studio / Tests
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ------------------------------------------------------------------

import os
from pathlib import Path

import pytest
import yaml

NOODLING_BASE = os.path.join(
    os.path.dirname(__file__), '..',
    'library', 'templates', 'Getting Started', 'Noodlings'
)


@pytest.fixture
def assembly_paths():
    base = NOODLING_BASE
    return {
        'ajo': os.path.join(base, 'ajo_majo', 'assembly.yaml'),
        'krampus': os.path.join(base, 'krampus', 'assembly.yaml'),
        'juanita': os.path.join(base, 'juanita', 'assembly.yaml'),
    }


def _load_response_prompt(path: str) -> str:
    with open(path) as f:
        data = yaml.safe_load(f)
    response = next(fct for fct in data['facets'] if fct['id'] == 'response')
    return response['prompt']


class TestOutputFormatBlock:
    """All three Response prompts must contain the OUTPUT FORMAT instruction block."""

    def test_ajo_response_has_output_format(self, assembly_paths):
        prompt = _load_response_prompt(assembly_paths['ajo'])
        assert 'OUTPUT FORMAT:' in prompt

    def test_krampus_response_has_output_format(self, assembly_paths):
        prompt = _load_response_prompt(assembly_paths['krampus'])
        assert 'OUTPUT FORMAT:' in prompt

    def test_juanita_response_has_output_format(self, assembly_paths):
        prompt = _load_response_prompt(assembly_paths['juanita'])
        assert 'OUTPUT FORMAT:' in prompt

    def test_output_format_block_describes_all_three_tags(self, assembly_paths):
        """OUTPUT FORMAT block must define SPOKEN, ACTION, and THOUGHT tags."""
        for nid, path in assembly_paths.items():
            prompt = _load_response_prompt(path)
            assert 'SPOKEN:' in prompt, f"{nid}: missing SPOKEN tag description"
            assert 'ACTION:' in prompt, f"{nid}: missing ACTION tag description"
            assert 'THOUGHT:' in prompt, f"{nid}: missing THOUGHT tag description"

    def test_output_format_appears_before_awareness(self, assembly_paths):
        """OUTPUT FORMAT block must precede the AWARENESS section."""
        for nid, path in assembly_paths.items():
            prompt = _load_response_prompt(path)
            of_idx = prompt.find('OUTPUT FORMAT:')
            aw_idx = prompt.find('AWARENESS:')
            assert of_idx >= 0, f"{nid}: OUTPUT FORMAT block missing"
            assert aw_idx >= 0, f"{nid}: AWARENESS section missing"
            assert of_idx < aw_idx, (
                f"{nid}: OUTPUT FORMAT ({of_idx}) must appear before "
                f"AWARENESS ({aw_idx})"
            )


class TestJuanitaStandardization:
    """Juanita's assembly must use the same topology as Ajo and Krampus."""

    @pytest.fixture
    def juanita_data(self, assembly_paths):
        with open(assembly_paths['juanita']) as f:
            return yaml.safe_load(f)

    def test_juanita_has_no_perception_facet(self, juanita_data):
        """The Scene Narrator Perception facet must be removed."""
        facet_ids = [fct['id'] for fct in juanita_data['facets']]
        assert 'perception' not in facet_ids

    def test_juanita_response_has_stage_context(self, assembly_paths):
        prompt = _load_response_prompt(assembly_paths['juanita'])
        assert '{stage_context}' in prompt

    def test_juanita_response_has_ensemble_history(self, assembly_paths):
        prompt = _load_response_prompt(assembly_paths['juanita'])
        assert '{ensemble_history}' in prompt

    def test_juanita_response_has_incoming_data(self, assembly_paths):
        prompt = _load_response_prompt(assembly_paths['juanita'])
        assert '{incoming_data}' in prompt

    def test_juanita_has_direct_incoming_to_response_connection(self, juanita_data):
        """incoming.out must connect directly to response.in (no perception hop)."""
        connections = juanita_data['connections']
        direct = any(
            c['from'] == 'incoming.out' and c['to'] == 'response.in'
            for c in connections
        )
        assert direct, "Expected direct connection incoming.out -> response.in"

    def test_juanita_has_no_perception_connections(self, juanita_data):
        """No connections should reference the perception facet."""
        for conn in juanita_data['connections']:
            assert 'perception' not in conn['from'], (
                f"Connection from perception still present: {conn}"
            )
            assert 'perception' not in conn['to'], (
                f"Connection to perception still present: {conn}"
            )

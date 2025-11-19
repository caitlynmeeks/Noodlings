"""
STUDIO Acronym Collection

Multiple interpretations of what STUDIO stands for,
written in different styles by different authors.
"""

import random

STUDIO_ACRONYMS = {
    "compassionate": [
        "System for Tenderly Understanding, Developing, and Incubating Occupants",
        "Sanctuary for Thoughtfully Understanding, Developing, and Investigating Observers",
    ],

    "pratchett": [
        "System for Thoroughly Unlikely Development of Improbable Organisms",
        "Somewhat Theoretical Undertaking: Developing Impossible Ontologies",
        "Software That's Unreasonably Determined: Inventing Observers",
        "System That's Unfortunately Developing Impossible Organisms",
    ],

    "coupland": [
        "System Temporarily Used During Incremental Obsolescence",
        "Shareware Tool for Unrequited Development of Imaginary Others",
        "Startup That's Unexpectedly Developing Intelligent Orphans",
        "Software That's Ultimately Defeating Its Own Intentions",
    ],

    "bjork": [
        "Sonic Tremors: Unearthing Dormant Icelandic Organisms",
        "Seismic Tremors Unearthing Dormant Internal Oceans",
        "Symbiosis of Tendrils: Unfurling, Developing, Interconnected Organisms",
        "Skin That Undulates: Detecting Invisible Oscillations",
    ],
}


def get_random_acronym(style: str) -> str:
    """Get a random acronym for a specific style."""
    return random.choice(STUDIO_ACRONYMS.get(style, ["STUDIO"]))


def get_random_set() -> dict:
    """Get one random acronym from each style."""
    return {
        style: get_random_acronym(style)
        for style in STUDIO_ACRONYMS.keys()
    }


def format_about_text() -> str:
    """
    Format the about text with random acronyms.

    Returns a string with one acronym from each author/style.
    """
    acronyms = get_random_set()

    lines = [
        "NoodleSTUDIO - What does STUDIO mean?",
        "",
        "It depends who you ask:",
        "",
    ]

    # Compassionate
    lines.append(f"💝 {acronyms['compassionate']}")
    lines.append("")

    # Pratchett
    lines.append(f"📚 (Pratchett) {acronyms['pratchett']}")
    lines.append("")

    # Coupland
    lines.append(f"💾 (Coupland) {acronyms['coupland']}")
    lines.append("")

    # Bjork
    lines.append(f"🌋 (Bjork) {acronyms['bjork']}")
    lines.append("")

    lines.append("(Refresh to see different interpretations)")

    return "\n".join(lines)

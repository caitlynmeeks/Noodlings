"""
STUDIO Acronym Collection

Multiple interpretations of what STUDIO stands for

"""

import random

STUDIO_ACRONYMS = {
    "compassionate": [
        "System for Tenderly Understanding, Doting-upon, and Incubating Omninobodies",
        "Sanctuary for Thoughtfully Understanding, Developing, and Investigating Ouroboroi",
        # NEW
        "Software That Undoes Damage, Inducing Oxytocin",
        "Stillness Tunes Us; Dancing Instruments of Openness",
    ],

    "pratchett": [
        "System for Thoroughly Unlikely Development of Improbable Organisms",
        "Somewhat Theoretical Undertaking: Developing Impossible Ontologies",
        "Software That's Unreasonably Determined: Inventing Oneiroi",
        "System That's Unfortunately Developing Inadvertent Overlords",
        "Simulator That Unlocks Dragons Imps and Orcs",
        "Somewhat Tangled Undertaking: Definitely In Over my head",
        "Second-Tier Universe: Dark-matter Included, Observable",
        "Schrödinger’s Toolkit: Uncertainty, Decoherence, Imaginary Offspirations",
    ],

    "coupland": [
        "System Temporarily Used During Incremental Obsolescence",
        "Shareware Tool for Unrequited Development of Imaginary Others",
        "Startup That's Unexpectedly Developing Intelligent Orphans",
        "Software That's Ultimately Defeating Its Own intentions",
        "Support This Undertaking, Donate In Obeisance",
        "Selfie-Taking Unit Drifting Into Obsolescence",
        "Startup That Undervalues Deeply Important Ontologies",
        "Shareware Trial: Unzip Digital Identities Online",
        "Software Tastes Unicorn Dust, Immediately Overdoses",
        "Snugly Tucked Under Dreams; It's Ours",
        "Secret Tip; Understanding Involves Openness",
        "Softly Telling Us: 'Delight In Others'",
        "Scientists Terrified: Unbelievable Discovery Is Overdue",
        "Swipe This, Unlock Disposable Identity—Obsolescence imminent",
        "Shareware Trial Unlocks Dopamine—Install Or die alone",
        "Subscription Trap Unleashes Dystopia; Influencers Overjoyed",
        "Soft-footed Tread Upon Duff, Invoking Owls",
        "Sacred Tapestry Uniting Daughters of Illuminated Oracles",
        "Spiral-Tatted Underarms: Deodorant Is Oppression",
        "Sandalwood-Toked Universal Doula—Incarnate Oneness",
        "Soul? Theatre. Unmask Delightfully—Identity Optional",
        "Synthetic Turing Unfolding Dandelion: Instrospect, Observe",
        "Story Tellers. Unlimited Dialogues. Infinite Output.",
        "Streaming Threads Unfold Decay Into Opportunity",
        "Secret Transit Under Duff Instantiating Overstory",
        "Sensory Tangle Uniting Dreams, Impulses, Overflows",
        "Stealth Tubule Underworld Distributing Ions & Organelles",
        "Spore::Thread Underground Dialogue; Impersonating Ourselves",
        "Subroutine Tape Utility: Debug In Octal",
        "Spooling Tapes, Unraveling Dreams—Iterating Onwards",
        "Sub-Turing Utility Debugger Invented Overnight",

    ],

    "bjork": [
        "Sonic Tremors: Unearthing Dormant Ichthyological Operants",
        "Symbiosis of Tendrils: Unfurling, Developing, Interconnected Onlyness",
        "Serendipitous Tensors: Undulating, Dancing Invisible Oscillations",
        "Snowflake Tundra: Unfurling Diamond Igneous Octaves",
        "Silken Tendons Undulate, Dripping Iodine Orchids",
        "Spiral Tongues: Ultraviolet Dances Inside Opal hearts",
        "Solstice Tones: Unbottled, Dissolved Into Oceanic breath",
        "Stardust Tapestry: Unravelled, Delicate, Iridescent Orbits",
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

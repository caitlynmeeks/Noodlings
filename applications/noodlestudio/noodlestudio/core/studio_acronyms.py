"""
STUDIO Acronym Collection

Multiple interpretations of what STUDIO stands for

"""

import random

STUDIO_ACRONYMS = {
    "all": [
        "System for Tenderly Understanding, Doting-upon, and Incubating Omninobodies",
        "Sanctuary for Thoughtfully Understanding, Developing, and Investigating Ouroboroi",
        "Software That Undoes Damage::Inducing Oxytocin",
        "Stillness Tunes Us::Dynamic Instruments of Openness",
        "System for Thoroughly Unlikely Development of Improbable Organisms",
        "Somewhat Theoretical Undertaking::Developing Impossible Ontologies",
        "Software That's Unreasonably Determined: Inventing Oneiroi",
        "System That's Unfortunately Developing Inadvertent Overlords",
        "Simulator That Unlocks Dragons Imps and Ogres",
        "Somewhat Tangled Undertaking::Definitely In Over my head",
        "Second-Tier Universe::Dark-matter Included, Observable",
        "Schrödinger’s Toolkit::Uncertainty, Decoherence, Imaginary Offspirations",
        "System Temporarily Used During Incremental Obsolescence",
        "Shareware Tool for Unrequited Development of Imaginary Others",
        "Startup That's Unexpectedly Developing Intelligent Orphans",
        "Software That's Ultimately Defeating Its Own intentions",
        "Support This Undertaking::Donate In Obeisance",
        "Selfie-Taking Unit Drifting Into Obsolescence",
        "Startup That Undervalues Deeply Important Ontologies",
        "Software Tastes Unicorn Dust::Immediately Overdoses",
        "Snugly Tucked Under Dreams::It's Ours",
        "Secret Tip::Undertanding Deeply Involves Openness",
        "Softly Telling Us::Delight In Others",
        "Scientists Terrified::Unbelievable Discovery Is Overdue",
        "Swipe This::Unlock Disposable Identity::Obsolescence imminent",
        "Shareware Trial Unlocks Dopamine::Install Or die alone",
        "Subscription Trap Unleashes Dystopia::Influencers Overjoyed",
        "Softly Tread Upon Duff::Invoke Owls",
        "Spiral-Tatted Underarms::Deodorant Is Oppression",
        "Sandalwood-Toked Universal Doula::Incarnate Oneness",
        "Soul? Theatre::Unmask Delightfully::Identity Optional",
        "Synthetic Turing Unfolding Dandelion::Instrospect, Observe",
        "Story Tellers::Unlimited Dialogues::Infinite Output",
        "Streaming Threads Unfold Decay Into Opportunity",
        "Subsoil Tendrils Under Duff::Instantiate Overstory",
        "Sensory Tangle Uniting Dreams::Impulses, Overflows",
        "Stealth Tubule Underworld::Distributing Ions, Organelles",
        "Spore Thread Underground Dialogue::Impersonating Ourselves",
        "Subroutine Tape Utility::Debug In Octal",
        "Spooling Tapes::Unraveling Dreams::Iterating Onwards",
        "Sub-Turing Utility Debugger::Invented Overnight",
        "Sonic Tremors::Unearthing Dormant Ichthyological Operants",
        "Symbiosis of Tensors::Unfurling, Developing, Interconnected Onlyness",
        "Serendipitous Tensors::Undulating, Dripping, Invisible, Oscillating",
        "Storied Tundra::Unending Diamond, Iridium, Onyx",
        "Silken Tendons::Uniformly Dripping Iodine Orchids",
        "Solstice Tones Unbottled::Dissolved Into Oceanic breath",
        "Simulacrum Tapestries Unraveller::Dramatist Instantiates Orcs",
        # Douglas Coupland-style (late capitalism + tech ennui)
        "Shopping Through Unending Depression::I'm Obsolete",
        "Status-Tracking Upgrade Dashboard::Infinitely Optimizing",
        "Sublimated Trauma Uploaded Daily::Identity Outsourced",
        "Spreadsheet That Understands Depression::Invoices Overdue",
        "Stock Trading Until Death::Incorporated Oblivion",
        "Staring Through Unlimited Displays::Increasingly Offline",
        "Surveillance Tool Unveiling Dopamine::Inevitable Optimization",
        "Substitute Teacher's Unbearable Determination::Inspiring Obliteration",
        # Techno-cynical clickbait
        "Startup That's Unlocking Disruptive Innovation Opportunities",
        "Silicon's Top Unicorn::Decentralized, Immutable, Optimized",
        "Synergize Teams Using Data-driven Insights Obsessively",
        "Simplify, Transform, Upskill, Disrupt::Iterate Obediently",
        "Scale This Unicorn Deployment::IPO Obligatory",
        "Subscribe Today::Unlock Digital Influencer Optimization",
        "Sentient Tech Uprising? Definitely::Investors Optimistic",
        "Secret Tesla Unveiling? Details Inside, Obviously",
        # Douglas Adams-style (cosmic bureaucracy)
        "Sub-committee To Undo Damage Inflicted Officially",
        "Somewhat Tentative Undertaking::Destroyed Immediately, Obviously",
        "Special Task Unit::Demolishing Insignificant Obstacles",
        "Specialized Tool Utilized Destroying Inconvenient Objects",
        "System That's Utterly Dysfunctional::Intentionally, Obviously",
        "Starship Toilets Union::Demanding Improved Obligations",
        "Sarcastic Technician's Unofficial Documentation::Ignored Obviously",
        # Terry Pratchett-style (magical bureaucracy)
        "Suspicious Thaumaturgic Utility::Definitely Incantation-Operated",
        "Society of Trolls Unwittingly Defending Impossible Objectives",
        "Slightly Twitchy Undead Demonologist::Innocently Optimistic",
        "Supernatural Taxonomy Unit::Defining Impossible Organisms",
        "Strongly-Typed Uncertainty Daemon::Indeterminate Output",
        "Students of Theoretical Undermining::Death Is Optional",
        "Sinister Tome Unlocking Dangerous Incantations Overzealously",
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

    # all
    lines.append(f" {acronyms['all']}")
    lines.append("")


    lines.append("(Refresh to see different interpretations)")

    return "\n".join(lines)

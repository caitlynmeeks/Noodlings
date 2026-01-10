# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#  ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Nonverbal Communication Formatters
#
#   Translates technical body language data into natural English.
#   Takes FACS codes (like AU6 = cheeks raised) and Laban movement
#   qualities (like "light + sudden") and describes them without
#   using emotion labels. Instead of saying "happy face," we say
#   "cheeks and corners lifted." This preserves the continuous
#   nature of affect rather than forcing discrete emotion boxes.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.cmush.nonverbal_formatters
# PURPOSE:  Convert FACS/Laban data to natural language descriptions
# LAYER:    Backend / Communication
# ──────────────────────────────────────────────────────────────
#
# KEY FUNCTIONS:
#   describe_facs()              FACS action units to text
#   describe_laban()             Laban effort qualities to text
#   format_nonverbal_for_chat()  Combined description for display
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: MIT
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

"""
Nonverbal Communication Formatters

Converts FACS (Facial Action Coding System) and Laban (body language)
JSON data into human-readable text descriptions using LLM.

NO DISCRETE EMOTION LABELS - preserves continuous affect space.

Author: Caitlyn + Claude
Date: November 25, 2025
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


async def describe_facs(facs_data: Dict[str, float], llm_client, affect_context: Dict[str, float] = None) -> str:
    """
    Convert FACS action units to human-readable facial expression using LLM.

    NO discrete emotion labels - LLM generates description from continuous AU data.

    Args:
        facs_data: Dict of AU codes to intensity values
                   e.g., {"AU6": 0.8, "AU12": 0.9}
        llm_client: LLM interface for generation
        affect_context: Optional continuous affect vector for context

    Returns:
        Human-readable description
        e.g., "*face with raised cheeks and pulled lip corners*"
    """
    if not facs_data:
        return "*neutral expression*"

    # Build AU description list (NO emotion labels, just muscle actions)
    au_descriptions = {
        "AU1": "inner brows raised",
        "AU2": "outer brows raised",
        "AU4": "brows lowered and drawn together",
        "AU5": "upper eyelids raised (eyes widened)",
        "AU6": "cheeks raised",
        "AU7": "eyelids tightened",
        "AU9": "nose wrinkled",
        "AU10": "upper lip raised",
        "AU12": "lip corners pulled obliquely upward",
        "AU15": "lip corners pulled down",
        "AU17": "chin raised",
        "AU20": "lips stretched horizontally",
        "AU23": "lips tightened",
        "AU25": "lips parted",
        "AU26": "jaw dropped",
        "AU27": "mouth stretched open"
    }

    # Filter to strong activations only
    active_aus = []
    for au, intensity in facs_data.items():
        if intensity > 0.5:
            desc = au_descriptions.get(au, f"{au} active")
            active_aus.append(f"{desc} ({intensity:.1f})")

    if not active_aus:
        return "*subtle facial movement*"

    # Build prompt for LLM
    aus_text = ", ".join(active_aus)

    affect_text = ""
    if affect_context:
        affect_text = f"\nCONTEXT - Continuous affect: valence={affect_context.get('valence', 0):.2f}, arousal={affect_context.get('arousal', 0):.2f}"

    prompt = f"""Describe this facial expression in natural language. Use ONLY the muscle actions listed - do NOT add emotion labels.

FACIAL ACTION UNITS:
{aus_text}
{affect_text}

STRICT RULES:
- Describe ONLY the physical muscle movements
- NO emotion words (happy, sad, angry, etc.)
- Keep it brief (3-7 words max)
- Format: *description*

Examples:
- AU6 + AU12 → "*cheeks and corners lifted*"
- AU1 + AU2 + AU5 → "*brows and eyes raised wide*"
- AU4 + AU7 → "*brows drawn, eyes tight*"

Your description:"""

    try:
        response = await llm_client.generate(
            prompt=prompt,
            system_prompt="You are a facial action descriptor. Describe muscle movements without emotion labels.",
            model='SMALL',
            max_tokens=30,
            temperature=0.5
        )
        return response.strip()
    except Exception as e:
        logger.warning(f"FACS LLM description failed: {e}, using fallback")
        # Fallback: just list the AUs
        return f"*{', '.join([au for au in facs_data.keys()])}*"


async def describe_laban(laban_data: Dict[str, str], llm_client, affect_context: Dict[str, float] = None) -> str:
    """
    Convert Laban effort qualities to human-readable body language using LLM.

    NO discrete emotion labels - LLM generates description from continuous effort data.

    Args:
        laban_data: Dict of effort dimensions
                    e.g., {"weight": "light", "time": "sudden", "space": "direct", "flow": "free"}
        llm_client: LLM interface for generation
        affect_context: Optional continuous affect vector for context

    Returns:
        Human-readable description
        e.g., "*light, quick movements*"
    """
    if not laban_data:
        return "*still*"

    # Build effort quality description (NO emotion labels)
    effort_text = []
    if laban_data.get("weight"):
        effort_text.append(f"Weight: {laban_data['weight']}")
    if laban_data.get("time"):
        effort_text.append(f"Time: {laban_data['time']}")
    if laban_data.get("space"):
        effort_text.append(f"Space: {laban_data['space']}")
    if laban_data.get("flow"):
        effort_text.append(f"Flow: {laban_data['flow']}")

    if not effort_text:
        return "*neutral posture*"

    affect_text = ""
    if affect_context:
        affect_text = f"\nCONTEXT - Continuous affect: valence={affect_context.get('valence', 0):.2f}, arousal={affect_context.get('arousal', 0):.2f}"

    prompt = f"""Describe this body language in natural language. Use ONLY the effort qualities listed - do NOT add emotion labels.

LABAN EFFORT QUALITIES:
{', '.join(effort_text)}
{affect_text}

STRICT RULES:
- Describe ONLY the movement qualities (weight, time, space, flow)
- NO emotion words (happy, sad, anxious, etc.)
- Keep it brief (3-7 words max)
- Format: *description*

Examples:
- light + sudden → "*quick, delicate movements*"
- strong + sustained → "*slow, powerful shifts*"
- indirect + bound → "*meandering, controlled motions*"

Your description:"""

    try:
        response = await llm_client.generate(
            prompt=prompt,
            system_prompt="You are a movement descriptor. Describe effort qualities without emotion labels.",
            model='SMALL',
            max_tokens=30,
            temperature=0.5
        )
        return response.strip()
    except Exception as e:
        logger.warning(f"Laban LLM description failed: {e}, using fallback")
        # Fallback: just list the qualities
        return f"*{laban_data.get('weight', '')}, {laban_data.get('time', '')} movements*"


async def format_nonverbal_for_chat(
    facs_data: Optional[Dict],
    laban_data: Optional[Dict],
    llm_client,
    affect_context: Optional[Dict] = None
) -> str:
    """
    Format FACS and Laban data for chat display using LLM.

    Args:
        facs_data: FACS action units
        laban_data: Laban effort qualities
        llm_client: LLM interface
        affect_context: Optional affect vector for context

    Returns:
        Formatted string for chat
        e.g., "*cheeks raised, quick movements*"
    """
    parts = []

    if facs_data:
        face = await describe_facs(facs_data, llm_client, affect_context)
        if face and face != "*neutral expression*":
            # Remove asterisks for combination
            parts.append(face.strip("*"))

    if laban_data:
        body = await describe_laban(laban_data, llm_client, affect_context)
        if body and body not in ["*still*", "*neutral posture*"]:
            # Remove asterisks for combination
            parts.append(body.strip("*"))

    if not parts:
        return ""

    return f"*{', '.join(parts)}*"


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    print("=== Nonverbal Formatters ===")
    print("Uses LLM to generate descriptions from FACS/Laban data")
    print("NO discrete emotion labels - preserves continuous affect space")
    print("\nTest by importing and calling with LLM client:")
    print("  face_desc = await describe_facs(facs_data, llm_client, affect)")
    print("  body_desc = await describe_laban(laban_data, llm_client, affect)")
    print("  combined = await format_nonverbal_for_chat(facs, laban, llm, affect)")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026

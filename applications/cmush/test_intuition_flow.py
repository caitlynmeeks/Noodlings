#!/usr/bin/env python3
"""
Test script to verify intuition flows through cognitive transistors correctly.

This simulates the cognition cycle to diagnose where intuition gets lost.
"""

import asyncio
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Simulate the data flow
async def simulate_intuition_flow():
    """Simulate how intuition flows through the system."""

    print("="*70)
    print("INTUITION FLOW DIAGNOSTIC")
    print("="*70)

    # Step 1: _generate_intuition() is called
    intuition_text = "That greeting is for Red, not me. Red is near the flames."
    print(f"\n1. _generate_intuition() returns:")
    print(f"   '{intuition_text}'")

    # Step 2: get_cognitive_transistor() retrieves IntuitionTransistor
    print(f"\n2. get_cognitive_transistor('IntuitionTransistor')")
    print(f"   -> Returns transistor instance")

    # Step 3: set_intuition() is called
    print(f"\n3. intuition_transistor.set_intuition(intuition_text)")
    print(f"   -> transistor.intuition_text = '{intuition_text}'")

    # Step 4: Context dict is built
    context = {
        'intuition': intuition_text,
        'affect': [0.5, 0.6, 0.2, 0.1, 0.3],
        'response_decision': {
            'response_type': 'think',
            'guidance': 'internal observation'
        }
    }
    print(f"\n4. Context dict built:")
    print(f"   context['intuition'] = '{intuition_text}'")
    print(f"   context['response_decision'] = {context['response_decision']}")

    # Step 5: fill_all_registers() is called
    print(f"\n5. cognitive_manifold.fill_all_registers(text, context, cycle_id)")

    # Step 6: For each transistor, fill_register() is called
    print(f"\n6. For IntuitionTransistor:")
    print(f"   transistor.fill_register(text, context, cycle_id)")
    print(f"     -> Calls transistor.process(text, context)")

    # Step 7: IntuitionTransistor.process() runs
    print(f"\n7. IntuitionTransistor.process() logic:")
    print(f"   - Checks: if not self.intuition_text:")
    print(f"   - self.intuition_text = '{intuition_text}'")
    print(f"   - Result: Has text, so process continues!")

    # Step 8: Build transformation prompt
    print(f"\n8. IntuitionTransistor builds prompt:")
    prompt_snippet = f"""INTUITIVE AWARENESS:
{intuition_text}

PERCEPTION: "hi red"

RESPONSE GUIDANCE:
You've decided to THINK: internal observation

Generate brief (1-2 sentences) content for this think..."""
    print(f"   {prompt_snippet[:200]}...")

    # Step 9: LLM transforms
    llm_output = "I sense Red is being addressed, not me."
    print(f"\n9. LLM returns:")
    print(f"   '{llm_output}'")

    # Step 10: TransistorOutput created
    print(f"\n10. Returns TransistorOutput:")
    print(f"    transformed_text = '{llm_output}'")
    print(f"    salience = 0.75")
    print(f"    metadata = {{'intuition': '{intuition_text}'}}")

    # Step 11: Register filled
    print(f"\n11. Register filled:")
    print(f"    transistor.register_state = 'ready'")
    print(f"    transistor.register_output = TransistorOutput(...)")

    # Step 12: All registers fill
    print(f"\n12. All transistors fill in parallel...")
    print(f"    IntuitionTransistor: READY")
    print(f"    AffectTransistor: READY")
    print(f"    PersonalityTransistor: READY")
    print(f"    ... etc")

    # Step 13: Pull lever
    print(f"\n13. integrate_from_registers():")
    print(f"    - Collects outputs from all registers")
    print(f"    - Blends them using LLM")

    # Step 14: Final output
    final_output = "I sense Red is being greeted. My flames feel curious but calm."
    print(f"\n14. Final manifold output:")
    print(f"    '{final_output}'")

    print("\n" + "="*70)
    print("EXPECTED RESULT: Intuition appears in NoodleTuner")
    print("="*70)

    print("\nCHECK in NoodleTuner:")
    print("  [IntuitionTransistor]")
    print("  Output: 'I sense Red is being addressed, not me.'")
    print("  Instruction Prompt: (should show full prompt with intuition)")
    print()

    # Now check what could go wrong
    print("="*70)
    print("POTENTIAL FAILURE POINTS")
    print("="*70)

    print("\n1. IF intuition_text is None or '':")
    print("   -> IntuitionTransistor.process() returns early")
    print("   -> TransistorOutput('', 0.1, {}) with low salience")
    print("   -> Shows blank in NoodleTuner")

    print("\n2. IF get_cognitive_transistor() returns wrong instance:")
    print("   -> set_intuition() updates wrong transistor")
    print("   -> process() uses transistor without intuition_text set")

    print("\n3. IF response_decision not in context:")
    print("   -> Prompt won't include 'You've decided to THINK'")
    print("   -> Still works, but less guidance")

    print("\n4. IF LLM fails in process():")
    print("   -> Falls back to: 'I sense: {intuition[:80]}'")
    print("   -> Should still show something")

    print("\n" + "="*70)
    print("DIAGNOSTIC STEPS")
    print("="*70)

    print("\n1. Check agent_bridge.py logs for:")
    print("   '[agent_id] Intuition generated: ...'")
    print("   '[agent_id] Updated IntuitionTransistor with: ...'")

    print("\n2. Check cognitive_components.py logs for:")
    print("   '[IntuitionTransistor] register READY (cycle xxx)'")

    print("\n3. In NoodleTuner, check IntuitionTransistor card:")
    print("   - Register State: should be 'ready' or 'computing'")
    print("   - Output: should show transformed text")
    print("   - Instruction Prompt: should show full prompt with intuition")

    print("\n4. Check manifold blend prompt includes intuition output")

    print("\n" + "="*70)

if __name__ == '__main__':
    asyncio.run(simulate_intuition_flow())

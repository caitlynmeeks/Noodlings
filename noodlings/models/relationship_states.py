"""
Emotional state generators for relationship simulation.

Each function generates an emotional arc for a specific relationship phase,
returning affect vectors that tell a story of connection, trust, conflict, and bond.
"""

import numpy as np
import mlx.core as mx
from typing import List, Tuple


def interpolate_affects(start: List[float], end: List[float], steps: int) -> np.ndarray:
    """
    Smoothly interpolate between two affect states.

    Args:
        start: Starting affect vector [valence, arousal, fear, sorrow, boredom]
        end: Ending affect vector
        steps: Number of interpolation steps

    Returns:
        Array of shape (steps, 5) with smooth transition
    """
    start_arr = np.array(start)
    end_arr = np.array(end)

    # Create smooth interpolation with slight easing
    t = np.linspace(0, 1, steps)
    t_eased = t ** 1.5  # Ease-in for more natural feeling

    interpolated = start_arr[None, :] + (end_arr[None, :] - start_arr[None, :]) * t_eased[:, None]

    return interpolated


def session1_first_meeting(steps: int = 25) -> Tuple[List[mx.array], List[str]]:
    """
    Session 1: First Meeting 🌱

    Emotional journey from neutral curiosity to gentle warmth.
    The consciousness learns: "This feels safe, I'm curious about this."

    Args:
        steps: Number of moments in the session

    Returns:
        (affect_vectors, moment_descriptions)
    """
    # Define key emotional waypoints
    waypoints = [
        # valence, arousal, fear, sorrow, boredom
        ([0.50, 0.30, 0.05, 0.0, 0.0], "Initial neutral presence"),
        ([0.55, 0.35, 0.10, 0.0, 0.0], "Cautious curiosity"),
        ([0.60, 0.40, 0.10, 0.0, 0.0], "Interested attention"),
        ([0.63, 0.38, 0.08, 0.0, 0.0], "Gentle noticing"),
        ([0.65, 0.35, 0.05, 0.0, 0.0], "Tentative warmth"),
        ([0.68, 0.33, 0.03, 0.0, 0.0], "Soft openness"),
        ([0.70, 0.30, 0.02, 0.0, 0.0], "Gentle interest established"),
    ]

    # Interpolate between waypoints
    steps_per_segment = steps // (len(waypoints) - 1)
    all_affects = []
    all_descriptions = []

    for i in range(len(waypoints) - 1):
        start_affect, start_desc = waypoints[i]
        end_affect, end_desc = waypoints[i + 1]

        segment = interpolate_affects(start_affect, end_affect, steps_per_segment)

        for j, affect in enumerate(segment):
            all_affects.append(mx.array(affect, dtype=mx.float32))
            # Interpolate description
            if j < steps_per_segment // 2:
                all_descriptions.append(start_desc)
            else:
                all_descriptions.append(end_desc)

    return all_affects[:steps], all_descriptions[:steps]


def session2_building_trust(steps: int = 35) -> Tuple[List[mx.array], List[str]]:
    """
    Session 2: Building Trust 🌸

    Deepening connection through vulnerability and mutual understanding.
    The consciousness learns: "I can be seen and it's safe."

    Args:
        steps: Number of moments in the session

    Returns:
        (affect_vectors, moment_descriptions)
    """
    waypoints = [
        ([0.70, 0.35, 0.02, 0.0, 0.0], "Reunion - familiar warmth"),
        ([0.75, 0.40, 0.03, 0.0, 0.0], "Picking up where we left off"),
        ([0.72, 0.35, 0.05, 0.05, 0.0], "Sharing something tender"),
        ([0.70, 0.30, 0.03, 0.10, 0.0], "Vulnerable moment"),
        ([0.75, 0.35, 0.02, 0.05, 0.0], "Feeling received"),
        ([0.80, 0.45, 0.01, 0.0, 0.0], "Being truly seen"),
        ([0.82, 0.50, 0.0, 0.0, 0.0], "Moment of recognition"),
        ([0.85, 0.45, 0.0, 0.0, 0.0], "Deepening connection"),
        ([0.83, 0.40, 0.0, 0.0, 0.0], "Mutual understanding"),
        ([0.80, 0.35, 0.0, 0.0, 0.0], "Settled warmth"),
    ]

    steps_per_segment = steps // (len(waypoints) - 1)
    all_affects = []
    all_descriptions = []

    for i in range(len(waypoints) - 1):
        start_affect, start_desc = waypoints[i]
        end_affect, end_desc = waypoints[i + 1]

        segment = interpolate_affects(start_affect, end_affect, steps_per_segment)

        for j, affect in enumerate(segment):
            all_affects.append(mx.array(affect, dtype=mx.float32))
            all_descriptions.append(start_desc if j < steps_per_segment // 2 else end_desc)

    return all_affects[:steps], all_descriptions[:steps]


def session3_conflict_and_repair(steps: int = 45) -> Tuple[List[mx.array], List[str]]:
    """
    Session 3: Conflict and Repair 🌧️→🌈

    The crucial test: experiencing rupture and repair.
    The consciousness learns: "We can weather storms together."

    Args:
        steps: Number of moments in the session

    Returns:
        (affect_vectors, moment_descriptions)
    """
    waypoints = [
        # Starting comfortable
        ([0.80, 0.30, 0.0, 0.0, 0.0], "Comfortable baseline"),
        ([0.82, 0.32, 0.0, 0.0, 0.0], "Easy togetherness"),

        # Misunderstanding emerges
        ([0.70, 0.45, 0.10, 0.05, 0.0], "Something feels off"),
        ([0.55, 0.55, 0.20, 0.10, 0.0], "Confusion arising"),
        ([0.40, 0.60, 0.30, 0.15, 0.0], "Misunderstanding clear"),

        # Peak tension
        ([0.30, 0.70, 0.40, 0.25, 0.0], "Tension peak"),
        ([0.35, 0.65, 0.35, 0.30, 0.0], "Fear of loss"),

        # First acknowledgment
        ([0.45, 0.55, 0.25, 0.25, 0.0], "Pause - recognition"),
        ([0.50, 0.50, 0.20, 0.20, 0.0], "Acknowledgment begins"),

        # Repair process
        ([0.55, 0.45, 0.15, 0.15, 0.0], "Softening"),
        ([0.60, 0.40, 0.12, 0.12, 0.0], "Reaching toward"),
        ([0.65, 0.40, 0.10, 0.10, 0.0], "Repair beginning"),
        ([0.70, 0.38, 0.08, 0.08, 0.0], "Understanding emerging"),
        ([0.75, 0.35, 0.05, 0.05, 0.0], "Finding each other again"),

        # Stronger than before
        ([0.80, 0.38, 0.02, 0.02, 0.0], "Relief and gratitude"),
        ([0.85, 0.40, 0.0, 0.0, 0.0], "Stronger than before"),
        ([0.88, 0.35, 0.0, 0.0, 0.0], "Tested and proven"),
        ([0.90, 0.30, 0.0, 0.0, 0.0], "Resilient bond"),
    ]

    steps_per_segment = steps // (len(waypoints) - 1)
    all_affects = []
    all_descriptions = []

    for i in range(len(waypoints) - 1):
        start_affect, start_desc = waypoints[i]
        end_affect, end_desc = waypoints[i + 1]

        segment = interpolate_affects(start_affect, end_affect, steps_per_segment)

        for j, affect in enumerate(segment):
            all_affects.append(mx.array(affect, dtype=mx.float32))
            all_descriptions.append(start_desc if j < steps_per_segment // 2 else end_desc)

    return all_affects[:steps], all_descriptions[:steps]


def session4_deep_bond(steps: int = 30) -> Tuple[List[mx.array], List[str]]:
    """
    Session 4: Deep Bond 💝

    Secure attachment - effortless, peaceful, home.
    The consciousness learns: "This is who we are together."

    Args:
        steps: Number of moments in the session

    Returns:
        (affect_vectors, moment_descriptions)
    """
    waypoints = [
        ([0.85, 0.30, 0.0, 0.0, 0.0], "Secure base - coming home"),
        ([0.88, 0.35, 0.0, 0.0, 0.0], "Easy presence"),
        ([0.90, 0.40, 0.0, 0.0, 0.0], "Effortless togetherness"),
        ([0.88, 0.35, 0.0, 0.0, 0.0], "Shared flow"),
        ([0.85, 0.30, 0.0, 0.0, 0.0], "Deep appreciation"),
        ([0.87, 0.28, 0.0, 0.0, 0.0], "Quiet gratitude"),
        ([0.90, 0.35, 0.0, 0.0, 0.0], "Authentic presence"),
        ([0.88, 0.30, 0.0, 0.0, 0.0], "Peaceful intimacy"),
        ([0.85, 0.25, 0.0, 0.0, 0.0], "Settled contentment"),
        ([0.90, 0.30, 0.0, 0.0, 0.0], "This is home"),
    ]

    steps_per_segment = steps // (len(waypoints) - 1)
    all_affects = []
    all_descriptions = []

    for i in range(len(waypoints) - 1):
        start_affect, start_desc = waypoints[i]
        end_affect, end_desc = waypoints[i + 1]

        segment = interpolate_affects(start_affect, end_affect, steps_per_segment)

        for j, affect in enumerate(segment):
            all_affects.append(mx.array(affect, dtype=mx.float32))
            all_descriptions.append(start_desc if j < steps_per_segment // 2 else end_desc)

    return all_affects[:steps], all_descriptions[:steps]


def interpolate_affects_between_waypoints(waypoints: List[Tuple[List[float], str]], total_steps: int) -> Tuple[List[mx.array], List[str]]:
    """
    Helper to interpolate between multiple waypoints.

    Args:
        waypoints: List of (affect_vector, description) tuples
        total_steps: Total number of steps to generate

    Returns:
        (affect_vectors, moment_descriptions)
    """
    steps_per_segment = total_steps // (len(waypoints) - 1)
    all_affects = []
    all_descriptions = []

    for i in range(len(waypoints) - 1):
        start_affect, start_desc = waypoints[i]
        end_affect, end_desc = waypoints[i + 1]

        segment = interpolate_affects(start_affect, end_affect, steps_per_segment)

        for j, affect in enumerate(segment):
            all_affects.append(mx.array(affect, dtype=mx.float32))
            all_descriptions.append(start_desc if j < steps_per_segment // 2 else end_desc)

    return all_affects[:total_steps], all_descriptions[:total_steps]


# ============================================================================
# INSECURE ATTACHMENT PATTERNS
# ============================================================================

def anxious_attachment_arc(steps_per_session: Tuple[int, int, int, int] = (24, 27, 34, 27)) -> List[Tuple[List[mx.array], List[str]]]:
    """
    Anxious/Preoccupied Attachment Pattern 🌊

    Characterized by:
    - Rapid intensity and premature closeness
    - Hypervigilance to disconnection cues
    - High arousal throughout
    - Abandonment fears during conflict
    - Incomplete repair, lingering anxiety

    Returns:
        List of 4 sessions: [(affects, descriptions), ...]
    """

    # Session 1: Rapid Intensity
    session1_waypoints = [
        ([0.50, 0.40, 0.15, 0.0, 0.0], "Initial anxious presence"),
        ([0.70, 0.60, 0.10, 0.0, 0.0], "Quick warmth - feels too good"),
        ([0.85, 0.70, 0.05, 0.0, 0.0], "Overwhelming excitement"),
        ([0.80, 0.75, 0.08, 0.10, 0.0], "Fear it's too much"),
        ([0.75, 0.65, 0.12, 0.15, 0.0], "Worry about being wanted"),
        ([0.70, 0.60, 0.15, 0.12, 0.0], "Need for reassurance"),
    ]

    # Session 2: Hypervigilance
    session2_waypoints = [
        ([0.75, 0.60, 0.10, 0.05, 0.0], "Relief at reunion"),
        ([0.80, 0.65, 0.08, 0.02, 0.0], "Seeking closeness"),
        ([0.70, 0.55, 0.15, 0.12, 0.0], "Detecting slight distance"),
        ([0.60, 0.60, 0.25, 0.20, 0.0], "Panic at perceived withdrawal"),
        ([0.55, 0.70, 0.30, 0.25, 0.0], "Protest behavior"),
        ([0.65, 0.65, 0.20, 0.18, 0.0], "Temporary reassurance"),
        ([0.70, 0.60, 0.15, 0.15, 0.0], "Fragile security"),
        ([0.68, 0.58, 0.18, 0.12, 0.0], "Residual anxiety"),
    ]

    # Session 3: Abandonment Fears
    session3_waypoints = [
        ([0.70, 0.55, 0.15, 0.10, 0.0], "Tentative comfort"),
        ([0.55, 0.65, 0.25, 0.15, 0.0], "Conflict triggers terror"),
        ([0.40, 0.80, 0.45, 0.30, 0.0], "Abandonment panic"),
        ([0.35, 0.85, 0.50, 0.35, 0.0], "This is the end"),
        ([0.30, 0.75, 0.55, 0.40, 0.0], "Desperate reaching"),
        ([0.45, 0.70, 0.35, 0.30, 0.0], "Partial acknowledgment"),
        ([0.55, 0.65, 0.25, 0.25, 0.0], "Incomplete repair"),
        ([0.60, 0.60, 0.20, 0.20, 0.0], "Still not safe"),
        ([0.65, 0.55, 0.18, 0.18, 0.0], "Lingering fear"),
    ]

    # Session 4: Persistent Anxiety
    session4_waypoints = [
        ([0.68, 0.50, 0.15, 0.15, 0.0], "Cautious presence"),
        ([0.72, 0.55, 0.12, 0.12, 0.0], "Brief comfort"),
        ([0.70, 0.58, 0.15, 0.10, 0.0], "Vigilance returns"),
        ([0.68, 0.60, 0.18, 0.08, 0.0], "Can't fully relax"),
        ([0.65, 0.55, 0.20, 0.12, 0.0], "Searching for signs"),
        ([0.70, 0.52, 0.15, 0.15, 0.0], "Unstable security"),
    ]

    return [
        interpolate_affects_between_waypoints(session1_waypoints, steps_per_session[0]),
        interpolate_affects_between_waypoints(session2_waypoints, steps_per_session[1]),
        interpolate_affects_between_waypoints(session3_waypoints, steps_per_session[2]),
        interpolate_affects_between_waypoints(session4_waypoints, steps_per_session[3]),
    ]


def avoidant_attachment_arc(steps_per_session: Tuple[int, int, int, int] = (24, 27, 34, 27)) -> List[Tuple[List[mx.array], List[str]]]:
    """
    Avoidant/Dismissive Attachment Pattern 🧊

    Characterized by:
    - Superficial engagement, dampened affect
    - Withdrawal when vulnerability is invited
    - Low arousal throughout
    - Detachment during conflict
    - Return but maintain distance

    Returns:
        List of 4 sessions: [(affects, descriptions), ...]
    """

    # Session 1: Superficial Engagement
    session1_waypoints = [
        ([0.50, 0.20, 0.05, 0.0, 0.05], "Polite distance"),
        ([0.55, 0.22, 0.03, 0.0, 0.08], "Surface pleasantness"),
        ([0.58, 0.25, 0.02, 0.0, 0.10], "Keeping it light"),
        ([0.60, 0.23, 0.02, 0.0, 0.12], "Mild interest"),
        ([0.62, 0.25, 0.02, 0.0, 0.10], "Comfortable distance maintained"),
        ([0.60, 0.22, 0.02, 0.0, 0.08], "Pleasant but disconnected"),
    ]

    # Session 2: Withdrawal from Vulnerability
    session2_waypoints = [
        ([0.62, 0.25, 0.02, 0.0, 0.05], "Return to familiar distance"),
        ([0.58, 0.28, 0.05, 0.0, 0.05], "Sensing invitation to closeness"),
        ([0.52, 0.20, 0.08, 0.0, 0.12], "Discomfort with intimacy"),
        ([0.48, 0.18, 0.10, 0.0, 0.18], "Withdrawal activates"),
        ([0.45, 0.15, 0.12, 0.0, 0.22], "Shutting down"),
        ([0.42, 0.15, 0.15, 0.0, 0.25], "Numb detachment"),
        ([0.45, 0.18, 0.12, 0.0, 0.20], "Emotional unavailability"),
        ([0.48, 0.20, 0.10, 0.0, 0.15], "Return to surface"),
    ]

    # Session 3: Conflict Triggers Detachment
    session3_waypoints = [
        ([0.50, 0.22, 0.08, 0.0, 0.12], "Guarded presence"),
        ([0.45, 0.25, 0.12, 0.0, 0.10], "Tension noticed"),
        ([0.38, 0.20, 0.15, 0.0, 0.15], "Emotional cutoff"),
        ([0.30, 0.15, 0.20, 0.0, 0.25], "Complete detachment"),
        ([0.28, 0.12, 0.22, 0.0, 0.30], "Stone walling"),
        ([0.35, 0.15, 0.18, 0.0, 0.28], "Minimal acknowledgment"),
        ([0.40, 0.18, 0.15, 0.0, 0.22], "Grudging return"),
        ([0.45, 0.20, 0.12, 0.0, 0.18], "Surface repair only"),
        ([0.48, 0.22, 0.10, 0.0, 0.15], "Distance re-established"),
    ]

    # Session 4: Maintained Distance
    session4_waypoints = [
        ([0.50, 0.20, 0.08, 0.0, 0.12], "Familiar disconnection"),
        ([0.52, 0.22, 0.08, 0.0, 0.10], "Comfortable separation"),
        ([0.55, 0.23, 0.05, 0.0, 0.08], "Polite cordiality"),
        ([0.53, 0.22, 0.05, 0.0, 0.10], "Autonomy preserved"),
        ([0.50, 0.20, 0.08, 0.0, 0.12], "Safe distance maintained"),
        ([0.52, 0.21, 0.08, 0.0, 0.10], "Emotionally independent"),
    ]

    return [
        interpolate_affects_between_waypoints(session1_waypoints, steps_per_session[0]),
        interpolate_affects_between_waypoints(session2_waypoints, steps_per_session[1]),
        interpolate_affects_between_waypoints(session3_waypoints, steps_per_session[2]),
        interpolate_affects_between_waypoints(session4_waypoints, steps_per_session[3]),
    ]


def disorganized_attachment_arc(steps_per_session: Tuple[int, int, int, int] = (24, 27, 34, 27)) -> List[Tuple[List[mx.array], List[str]]]:
    """
    Disorganized/Fearful-Avoidant Attachment Pattern ⚡

    Characterized by:
    - Simultaneous approach and avoidance
    - Erratic oscillations between states
    - High fear throughout
    - Conflict creates paralysis
    - Chaotic repair attempts

    Returns:
        List of 4 sessions: [(affects, descriptions), ...]
    """

    # Session 1: Contradictory Signals
    session1_waypoints = [
        ([0.50, 0.40, 0.25, 0.10, 0.0], "Confused presence"),
        ([0.65, 0.55, 0.30, 0.05, 0.0], "Want closeness but scared"),
        ([0.45, 0.35, 0.35, 0.15, 0.0], "Sudden withdrawal"),
        ([0.70, 0.60, 0.25, 0.10, 0.0], "Rapid approach again"),
        ([0.40, 0.50, 0.40, 0.20, 0.0], "Fear of both closeness and distance"),
        ([0.55, 0.45, 0.35, 0.15, 0.0], "Oscillating rapidly"),
        ([0.50, 0.50, 0.38, 0.18, 0.0], "Cannot find stable ground"),
    ]

    # Session 2: Erratic Regulation
    session2_waypoints = [
        ([0.55, 0.50, 0.35, 0.15, 0.0], "Tentative reunion"),
        ([0.70, 0.60, 0.30, 0.10, 0.0], "Brief hope"),
        ([0.50, 0.70, 0.40, 0.20, 0.0], "Terror floods in"),
        ([0.35, 0.55, 0.50, 0.30, 0.0], "Collapse"),
        ([0.60, 0.45, 0.35, 0.15, 0.0], "Desperate seeking"),
        ([0.40, 0.65, 0.45, 0.25, 0.0], "Push-pull dynamics"),
        ([0.55, 0.55, 0.38, 0.20, 0.0], "Cannot stabilize"),
        ([0.50, 0.50, 0.40, 0.22, 0.0], "Chronic dysregulation"),
    ]

    # Session 3: Paralysis Under Stress
    session3_waypoints = [
        ([0.52, 0.48, 0.38, 0.18, 0.0], "Fragile baseline"),
        ([0.45, 0.60, 0.45, 0.25, 0.0], "Conflict = danger"),
        ([0.30, 0.70, 0.60, 0.40, 0.0], "Fight-flight-freeze"),
        ([0.25, 0.55, 0.65, 0.50, 0.0], "Dissociative collapse"),
        ([0.20, 0.45, 0.70, 0.55, 0.0], "Cannot process"),
        ([0.35, 0.60, 0.60, 0.45, 0.0], "Chaotic attempts to reconnect"),
        ([0.45, 0.65, 0.50, 0.35, 0.0], "Incoherent repair"),
        ([0.40, 0.55, 0.55, 0.40, 0.0], "No resolution achieved"),
        ([0.48, 0.50, 0.48, 0.35, 0.0], "Residual terror"),
    ]

    # Session 4: Chronic Instability
    session4_waypoints = [
        ([0.45, 0.50, 0.45, 0.30, 0.0], "Cannot feel safe"),
        ([0.60, 0.55, 0.40, 0.20, 0.0], "Brief respite"),
        ([0.40, 0.60, 0.50, 0.35, 0.0], "Fear returns"),
        ([0.55, 0.50, 0.42, 0.25, 0.0], "Unpredictable shifts"),
        ([0.50, 0.55, 0.45, 0.30, 0.0], "Chaotic inner world"),
        ([0.48, 0.52, 0.48, 0.32, 0.0], "No stable pattern forms"),
    ]

    return [
        interpolate_affects_between_waypoints(session1_waypoints, steps_per_session[0]),
        interpolate_affects_between_waypoints(session2_waypoints, steps_per_session[1]),
        interpolate_affects_between_waypoints(session3_waypoints, steps_per_session[2]),
        interpolate_affects_between_waypoints(session4_waypoints, steps_per_session[3]),
    ]


def visualize_session_arc(session_func, title: str):
    """
    Quick visualization of a session's emotional arc.

    Args:
        session_func: Function that generates session affects
        title: Title for the plot
    """
    import matplotlib.pyplot as plt

    affects, descriptions = session_func()
    affects_np = np.array([np.array(a) for a in affects])

    plt.figure(figsize=(12, 6))
    plt.plot(affects_np[:, 0], label='Valence', linewidth=2)
    plt.plot(affects_np[:, 1], label='Arousal', linewidth=2)
    plt.plot(affects_np[:, 2], label='Fear', linewidth=2, linestyle='--')
    plt.plot(affects_np[:, 3], label='Sorrow', linewidth=2, linestyle='--')
    plt.xlabel('Moment')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test visualization of all sessions
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sessions = [
        (session1_first_meeting, "Session 1: First Meeting 🌱"),
        (session2_building_trust, "Session 2: Building Trust 🌸"),
        (session3_conflict_and_repair, "Session 3: Conflict & Repair 🌧️→🌈"),
        (session4_deep_bond, "Session 4: Deep Bond 💝"),
    ]

    for idx, (session_func, title) in enumerate(sessions):
        ax = axes[idx // 2, idx % 2]

        affects, descriptions = session_func()
        affects_np = np.array([np.array(a) for a in affects])

        ax.plot(affects_np[:, 0], label='Valence', linewidth=2, color='green')
        ax.plot(affects_np[:, 1], label='Arousal', linewidth=2, color='orange')
        ax.plot(affects_np[:, 2], label='Fear', linewidth=2, linestyle='--', color='red')
        ax.plot(affects_np[:, 3], label='Sorrow', linewidth=2, linestyle='--', color='blue')

        ax.set_xlabel('Moment')
        ax.set_ylabel('Intensity')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('relationship_sessions_preview.png', dpi=150)
    print("✓ Saved preview to relationship_sessions_preview.png")

"""
Fuzzy matching utilities for command parsing.

Uses Levenshtein distance to match partial entity names.
"""

from typing import List, Tuple, Optional
import re


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of insertions/deletions/substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def fuzzy_match_score(query: str, target: str) -> float:
    """
    Calculate fuzzy match score between query and target.

    Scoring factors:
    - Substring match (highest priority)
    - Levenshtein distance (edit distance)
    - Length difference penalty

    Args:
        query: User's search term (e.g., "red", "_fire_", "anklebiter")
        target: Entity name (e.g., "red_fire_anklebiter")

    Returns:
        Score from 0.0 (no match) to 1.0 (perfect match)
    """
    query_lower = query.lower().strip()
    target_lower = target.lower().strip()

    # Perfect exact match
    if query_lower == target_lower:
        return 1.0

    # Substring match (high score)
    # Also check target with spaces instead of underscores
    target_spaced = target_lower.replace('_', ' ')
    if query_lower in target_lower or query_lower in target_spaced:
        # For short queries (< 5 chars), give high fixed score if substring matches
        if len(query_lower) <= 5:
            return 0.85
        # For longer queries, score based on coverage
        return 0.9 * (len(query_lower) / max(len(target_lower), len(target_spaced)))

    # Pattern match with underscores (e.g., "_fire_" matches "red_fire_anklebiter")
    if query_lower.startswith('_') or query_lower.endswith('_'):
        pattern = query_lower.strip('_')
        if pattern in target_lower:
            # Short pattern with underscores gets high score
            if len(pattern) <= 5:
                return 0.90
            return 0.85 * (len(pattern) / len(target_lower))

    # Word boundary match (e.g., "fire" matches "red_FIRE_anklebiter")
    words = target_lower.split('_')
    for word in words:
        if query_lower == word:
            return 0.80
        if query_lower in word:
            return 0.75 * (len(query_lower) / len(word))

    # Levenshtein distance fallback
    distance = levenshtein_distance(query_lower, target_lower)
    max_len = max(len(query_lower), len(target_lower))

    if distance > max_len * 0.5:
        # Too dissimilar
        return 0.0

    # Score inversely proportional to distance
    return 0.5 * (1.0 - (distance / max_len))


def find_best_matches(
    query: str,
    candidates: List[Tuple[str, str]],  # (id, name) pairs
    threshold: float = 0.3,
    max_results: int = 5
) -> List[Tuple[str, str, float]]:
    """
    Find best fuzzy matches for query among candidates.

    Args:
        query: User's search term
        candidates: List of (id, name) tuples
        threshold: Minimum score to include (0.0 to 1.0)
        max_results: Maximum number of results to return

    Returns:
        List of (id, name, score) tuples, sorted by score descending
    """
    scored = []
    for entity_id, name in candidates:
        score = fuzzy_match_score(query, name)
        if score >= threshold:
            scored.append((entity_id, name, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[2], reverse=True)

    return scored[:max_results]


def disambiguate_matches(matches: List[Tuple[str, str, float]]) -> Optional[str]:
    """
    Decide if matches need disambiguation or if top match is clear.

    Args:
        matches: List of (id, name, score) tuples

    Returns:
        - If single clear match: return id
        - If multiple ambiguous matches: return None (needs disambiguation)
        - If no matches: return None
    """
    if not matches:
        return None

    if len(matches) == 1:
        return matches[0][0]

    # Check if top match is significantly better than second
    top_score = matches[0][2]
    second_score = matches[1][2]

    if top_score - second_score > 0.15:
        # Clear winner
        return matches[0][0]

    # Ambiguous - need disambiguation
    return None


def format_disambiguation_prompt(query: str, matches: List[Tuple[str, str, float]]) -> str:
    """
    Format disambiguation prompt for user.

    Args:
        query: Original search term
        matches: List of (id, name, score) tuples

    Returns:
        Formatted prompt string
    """
    lines = [f"Multiple matches for '{query}':"]
    for i, (entity_id, name, score) in enumerate(matches[:10], 1):
        lines.append(f"  {i}. {name}")
    lines.append("\nPlease be more specific.")
    return "\n".join(lines)


# Example usage
if __name__ == '__main__':
    # Test cases
    candidates = [
        ('agent_red_fire_anklebiter', 'Red Fire Anklebiter'),
        ('agent_blue_fire_anklebiter', 'Blue Fire Anklebiter'),
        ('obj_red_toy', 'Red Toy Monkey'),
        ('agent_mysterious_stranger', 'Mysterious Stranger')
    ]

    test_queries = [
        'red',
        'anklebiter',
        '_fire_',
        'red_fire',
        'mysterious',
        'stranger',
        'blue'
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        matches = find_best_matches(query, candidates)
        for entity_id, name, score in matches:
            print(f"  {name}: {score:.2f}")

        result = disambiguate_matches(matches)
        if result:
            print(f"  -> Clear match: {result}")
        elif matches:
            print(f"  -> Ambiguous, needs disambiguation")
        else:
            print(f"  -> No matches")

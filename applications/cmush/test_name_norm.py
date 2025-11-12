#!/usr/bin/env python3
"""Quick test of name normalization"""

# Simulate the normalization
available_cast = ["toad", "phi"]
cast_lower_map = {name.lower(): name for name in available_cast}

test_names = [
    "Mr. Toad",
    "Mr.Toad",
    "Mr. To ad",  # with space in name
    "Phi",
    "phi"
]

print(f"available_cast = {available_cast}")
print(f"cast_lower_map = {cast_lower_map}")
print()

for cast_member in test_names:
    cast_lower = cast_member.strip().lower().replace("mr. ", "").replace("ms. ", "").replace("mrs. ", "").replace(".", "").strip()
    found = cast_lower in cast_lower_map
    print(f"cast_member='{cast_member}' -> cast_lower='{cast_lower}' -> found={found}")
    if found:
        print(f"  Normalized to: {cast_lower_map[cast_lower]}")
    print()

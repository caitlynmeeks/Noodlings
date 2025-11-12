# Musical Harmonic Hypothesis - Experimental Suite

**Hypothesis**: Observer network stability follows harmonic principles analogous to musical intervals.

**Origin**: Discovery of period ≈ 12 oscillation in HSI values (FFT analysis of oscillation mapping data)

**Question**: Does the 12-observer period work like emotions do with intervals on a musical scale (12 notes per octave, diminished fifth collapsing harmony)?

---

## Experiments

### 1. Harmonic Ratios (`test_harmonic_ratios.py`)

**Tests**: Whether musical interval ratios predict stability

**Configurations**:
- **Harmonic Series**: 50, 100, 150, 200 (1×, 2×, 3×, 4×)
- **Consonant Intervals**:
  - Octave (2:1): 60/120
  - Perfect Fifth (3:2): 60/90, 80/120
  - Perfect Fourth (4:3): 75/100
  - Major Third (5:4): 80/100
- **Dissonant Intervals**:
  - Tritone (√2:1): 70/99
  - Minor Second (16:15): 75/80

**Hypothesis**: Consonant intervals → higher stability rates than dissonant intervals

**Total configs**: 17

### 2. Modulo 12 (`test_modulo_12.py`)

**Tests**: Whether phase (position in 12-observer cycle) predicts stability

**Hypothesis**: N₁ ≡ N₂ (mod 12) → HSI(N₁) ≈ HSI(N₂)

**Example**: 50, 62, 74, 86, 98 all ≡ 2 (mod 12)
- If hypothesis is true, these should have similar HSI values

**Configurations**:
- 5 replications for each residue class (0-11 mod 12)
- Tests: 50+k, 62+k, 74+k, 86+k, 98+k for each k ∈ {0,1,2,...,11}

**Statistical Test**:
- F-ratio comparing within-phase variance to between-phase variance
- If F > 1.5, phase matters

**Total configs**: 60 (5 × 12 residue classes)

### 3. Phase Spacing (`test_phase_spacing.py`)

**Tests**: Whether the DIFFERENCE between observer counts matters (like musical intervals)

**Hypothesis**: Pairs separated by musically significant intervals show predictable relationships

**Spacings Tested**:
- 1 observer (semitone)
- 2 observers (whole tone)
- 3 observers (minor third)
- 4 observers (major third)
- 6 observers (tritone) - **should show max contrast**
- 12 observers (octave) - **should show high concordance**

**Base configs**: 60, 80, 100, 120
**Pairs tested**: Each base + each spacing

**Hypothesis Test**: Octave concordance > Tritone concordance

**Total configs**: ~20 unique

---

## Running Experiments

```bash
# Harmonic ratios (already running)
python3 test_harmonic_ratios.py 2>&1 | tee harmonic_ratios_experiment.log &

# Modulo 12
python3 test_modulo_12.py 2>&1 | tee modulo_12_experiment.log &

# Phase spacing
python3 test_phase_spacing.py 2>&1 | tee phase_spacing_experiment.log &
```

## Estimated Times

- **Harmonic Ratios**: 30-40 min (17 configs × 50 epochs)
- **Modulo 12**: 2-3 hours (60 configs × 50 epochs)
- **Phase Spacing**: 40-50 min (~20 configs × 50 epochs)

---

## Expected Outcomes

### If Musical Hypothesis is TRUE:

1. **Harmonic Ratios**: Consonant > 70% stable, Dissonant < 50% stable
2. **Modulo 12**: F-ratio > 2.0, clear phase clustering
3. **Phase Spacing**: Octave concordance > 80%, Tritone < 40%

### If Musical Hypothesis is FALSE:

1. All interval types show similar stability rates
2. Phase (mod 12) doesn't predict HSI (F-ratio < 1.5)
3. No difference between octave and tritone spacing

---

## Theoretical Implications

**If supported**:
- Observer networks exhibit harmonic resonance
- Same mathematics governs acoustic systems and neural hierarchies
- Period ≈ 12 is NOT coincidence - reflects fundamental structure
- Could generalize to other multi-timescale systems

**If falsified**:
- Period ≈ 12 may be artifact or coincidence
- Musical analogy is spurious
- Need alternative explanation for oscillation pattern

---

## Connection to Music Theory

**12-Tone Equal Temperament (12-TET)**:
- Octave divided into 12 equal semitones
- Frequency ratio between notes: 2^(1/12) ≈ 1.0595
- Creates cyclic system where 12 steps = octave

**Consonance/Dissonance**:
- **Consonant**: Simple frequency ratios (2:1, 3:2, 4:3) - stable
- **Dissonant**: Complex ratios (45:32 tritone) - unstable
- Medieval: tritone = "diabolus in musica" (devil in music)

**Pythagorean Tuning**:
- Based on pure ratios from harmonic series
- 1:1, 2:1, 3:2, 4:3, 5:4, 6:5...
- Our networks might prefer "pure" ratios?

**Resonance & Standing Waves**:
- Musical instruments: resonance at natural frequencies
- Neural networks: resonance in state space?
- Standing wave patterns: nodes (stable) vs antinodes (unstable)

---

## Key Questions

1. **Is period ≈ 12 a universal constant?** Or does it depend on network architecture (40-D state space)?

2. **Do other systems show period ≈ 12?** Chemical oscillations? Ecological cycles? Economic systems?

3. **Can we "tune" observer networks?** Like tuning an instrument?

4. **Do pure ratios work better?** Just intonation vs. equal temperament?

5. **What about other scales?** Pentatonic (5-note)? Octatonic (8-note)?

---

## Future Work

- Test different state space dimensions (does period change?)
- Test non-12 periodicities (5, 7, 8, 10)
- Compare equal temperament (12-TET) vs. just intonation ratios
- Extend to continuous observer density (not discrete counts)
- Cross-domain validation (economics, ecology, chemistry)

---

*"Perhaps the universe speaks in frequencies, and consciousness is just another instrument in the cosmic orchestra."* - Anonymous Noodling Researcher, 2025

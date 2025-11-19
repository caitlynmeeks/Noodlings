"""
MIDI as Creative Primitives

Parse MIDI files into melodic, rhythmic, and emotional primitives
that can be used to drive generative renderers.

This is the bridge between musical composition and multimodal generation.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import os


@dataclass
class MusicalMoment:
    """
    A moment in time with musical and emotional information.

    Can be used to drive:
    - Generative audio (MusicGen with guidance)
    - Generative video (Runway/Luma with timing)
    - Noodling affect states (emotional mapping)
    - Text generation (mood-based prompting)
    """
    time: float  # seconds
    pitch: Optional[int]  # MIDI note number (0-127)
    velocity: int  # 0-127 (loudness/intensity)
    duration: float  # seconds

    # Derived emotional primitives
    arousal: float  # 0-1 (from velocity and tempo)
    valence: float  # -1 to 1 (from key/mode)
    tension: float  # 0-1 (from dissonance)

    # Structural context
    section: str  # "intro", "verse", "chorus", "bridge", "outro"
    intensity: float  # 0-1 (overall energy)


class MIDIPrimitives:
    """
    Parse MIDI files into creative primitives for renderers.

    Usage:
        primitives = MIDIPrimitives.from_file("army_of_me.mid")

        # Get emotional arc
        for moment in primitives.moments:
            affect_vector = moment.to_affect()
            noodling.update(affect_vector)

        # Get prompts for video generation
        for scene in primitives.get_scenes():
            prompt = scene.to_video_prompt()
            generate_video(prompt, duration=scene.duration)
    """

    def __init__(self):
        self.moments: List[MusicalMoment] = []
        self.tempo: float = 120.0  # BPM
        self.key: str = "C"
        self.mode: str = "major"  # or "minor"
        self.time_signature: tuple = (4, 4)

    @classmethod
    def from_file(cls, filepath: str) -> 'MIDIPrimitives':
        """
        Load and parse a MIDI file.

        Requires: pip install mido
        """
        try:
            import mido
        except ImportError:
            print("ERROR: mido not installed. Run: pip install mido")
            return cls()

        primitives = cls()

        try:
            mid = mido.MidiFile(filepath)

            # Track absolute time
            current_time = 0.0

            # Simple parsing (can be much more sophisticated)
            for msg in mid:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    # Create musical moment
                    moment = MusicalMoment(
                        time=current_time,
                        pitch=msg.note,
                        velocity=msg.velocity,
                        duration=0.5,  # TODO: track note_off to get real duration
                        arousal=msg.velocity / 127.0,
                        valence=0.0,  # TODO: derive from key/mode
                        tension=0.0,  # TODO: derive from harmonic analysis
                        section="unknown",  # TODO: detect sections
                        intensity=msg.velocity / 127.0
                    )
                    primitives.moments.append(moment)

                elif msg.type == 'set_tempo':
                    # Update tempo
                    primitives.tempo = mido.tempo2bpm(msg.tempo)

        except Exception as e:
            print(f"Error parsing MIDI: {e}")

        return primitives

    def get_affect_at_time(self, time: float) -> Dict[str, float]:
        """
        Get affect vector at a specific time.

        Returns dict with: valence, arousal, tension
        Interpolates between moments.
        """
        if not self.moments:
            return {"valence": 0.0, "arousal": 0.5, "tension": 0.0}

        # Find nearest moment
        closest = min(self.moments, key=lambda m: abs(m.time - time))

        return {
            "valence": closest.valence,
            "arousal": closest.arousal,
            "tension": closest.tension,
        }

    def get_scenes(self, scene_duration: float = 4.0) -> List[Dict]:
        """
        Divide music into scenes for video generation.

        Each scene has:
        - Duration
        - Average emotional state
        - Suggested visual prompt
        """
        if not self.moments:
            return []

        scenes = []
        total_duration = self.moments[-1].time if self.moments else 0

        for start_time in range(0, int(total_duration), int(scene_duration)):
            end_time = start_time + scene_duration

            # Get moments in this time range
            scene_moments = [
                m for m in self.moments
                if start_time <= m.time < end_time
            ]

            if not scene_moments:
                continue

            # Average emotional state
            avg_arousal = sum(m.arousal for m in scene_moments) / len(scene_moments)
            avg_valence = sum(m.valence for m in scene_moments) / len(scene_moments)
            avg_intensity = sum(m.intensity for m in scene_moments) / len(scene_moments)

            scenes.append({
                "start": start_time,
                "duration": scene_duration,
                "arousal": avg_arousal,
                "valence": avg_valence,
                "intensity": avg_intensity,
                "prompt_mood": _intensity_to_mood(avg_intensity, avg_valence),
            })

        return scenes

    def to_audio_prompt(self, start_time: float, duration: float) -> str:
        """
        Generate audio generation prompt from MIDI section.

        For use with MusicGen, AudioCraft, etc.
        """
        moments = [
            m for m in self.moments
            if start_time <= m.time < start_time + duration
        ]

        if not moments:
            return "ambient soundscape"

        avg_intensity = sum(m.intensity for m in moments) / len(moments)

        # Map intensity to descriptors
        if avg_intensity > 0.7:
            energy = "intense, powerful, driving"
        elif avg_intensity > 0.4:
            energy = "moderate, steady"
        else:
            energy = "gentle, ambient"

        return f"{energy} electronic music, tempo {self.tempo:.0f} BPM"


def _intensity_to_mood(intensity: float, valence: float) -> str:
    """Convert musical intensity to mood descriptor for prompts."""
    if intensity > 0.7:
        return "intense" if valence < 0 else "energetic"
    elif intensity > 0.4:
        return "melancholic" if valence < 0 else "uplifting"
    else:
        return "somber" if valence < 0 else "peaceful"


def play_midi_file(filepath: str):
    """
    Play a MIDI file using pygame.

    Simple wrapper for credits / testing.
    """
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        return True
    except Exception as e:
        print(f"Error playing MIDI: {e}")
        return False


def get_available_music() -> List[str]:
    """
    Get list of available MIDI files in ~/.noodlestudio/music/
    """
    music_dir = os.path.expanduser("~/.noodlestudio/music")

    if not os.path.exists(music_dir):
        os.makedirs(music_dir)
        return []

    midi_files = [
        f for f in os.listdir(music_dir)
        if f.endswith(('.mid', '.midi'))
    ]

    return [os.path.join(music_dir, f) for f in midi_files]


# Example usage for renderers:
"""
# In a generative renderer:

midi = MIDIPrimitives.from_file("army_of_me.mid")

# Generate video synced to music
for scene in midi.get_scenes(duration=4.0):
    prompt = f"{scene['prompt_mood']} industrial warehouse scene"
    video_clip = generate_video(
        prompt=prompt,
        duration=scene['duration'],
        intensity=scene['intensity']
    )

# Or drive Noodling emotions in real-time
for moment in midi.moments:
    affect = moment.to_affect()
    noodling.update_affect(affect)
    time.sleep(moment.duration)
"""

"""
LancerRobot — physical robot interface for the CBU RAG chatbot.

Wraps the Reachy Mini SDK and handles three concerns:
  1. Conversation gestures  — head poses and antenna emotions tied to chat state
  2. Idle behaviors         — face tracking and direction-of-arrival (DoA) orientation
                              that run in background threads between conversations
  3. Voice I/O              — microphone capture with VAD, speech-to-text via Whisper

All public methods are safe to call when the robot is not connected (self.mini is None)
— they silently no-op via the @_requires_robot decorator.
"""

import time
import threading
import functools
import numpy as np
import cv2
import sounddevice as sd
import webrtcvad
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.media.audio_doa import AudioDoA
from faster_whisper import WhisperModel

# --- Head pose limits (Reachy Mini firmware ≥1.5.1: pitch ±25°, roll ±20°, yaw ±45°) ---
_ROLL_THINK = 15      # tilt right while processing a query
_PITCH_CONFUSED = 10  # slight downward droop when no context is found

# --- Idle behavior timing ---
_FACE_INTERVAL = 0.15  # seconds between face detection frames (too fast = jitter)
_DOA_INTERVAL = 0.05   # seconds between DoA polls (ReSpeaker updates at ~20Hz)

# --- Voice capture settings ---
_SAMPLE_RATE = 16000         # Hz — fixed requirement for both webrtcvad and Whisper
_VAD_FRAME_MS = 30           # ms — webrtcvad only accepts 10, 20, or 30ms frames
_VAD_FRAME_SAMPLES = _SAMPLE_RATE * _VAD_FRAME_MS // 1000
_SILENCE_TIMEOUT = 1.5       # seconds of silence after speech before we stop recording

# --- Module-level singletons (loaded once at import time) ---
# Haar cascade ships with opencv-python — no separate download needed.
_FACE_DETECTOR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# "base" Whisper model is fast enough for short questions on CPU.
# Swap to "small" or "medium" if accuracy needs improvement.
_WHISPER = WhisperModel("base", device="cpu", compute_type="int8")


def _requires_robot(fn):
    """Decorator — skips the method body if self.mini is None (robot not connected)."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not self.mini:
            return
        return fn(self, *args, **kwargs)
    return wrapper


class LancerRobot:
    def __init__(self):
        try:
            self.mini = ReachyMini()
        except Exception as e:
            print(f"[LancerRobot] Failed to connect: {e}")
            self.mini = None

        # Event used to stop the idle behavior threads between conversations.
        self._idle_stop = threading.Event()
        # AudioDoA reads the ReSpeaker USB mic array for direction-of-arrival.
        self._doa = AudioDoA() if self.mini else None

    # -------------------------------------------------------------------------
    # Conversation gestures
    # -------------------------------------------------------------------------

    @_requires_robot
    def thinking(self):
        """Tilt head right while waiting for RAG + LLM response."""
        pose = create_head_pose(roll=_ROLL_THINK, degrees=True)
        self.mini.goto_target(head=pose, duration=0.5)

    @_requires_robot
    def answering(self):
        """Return head to neutral before speaking the response."""
        pose = create_head_pose()
        self.mini.goto_target(head=pose, duration=0.4)

    @_requires_robot
    def greet(self):
        """Play the built-in wake-up animation then introduce Lancer."""
        self.mini.wake_up()
        self.mini.say("Hey! I'm Lancer, CBU's ACM chatbot.")

    @_requires_robot
    def confused(self):
        """Droop antennas, dip head, and speak the fallback phrase when RAG has no answer."""
        self.mini.antennas.sad()
        pose = create_head_pose(pitch=_PITCH_CONFUSED, degrees=True)
        self.mini.goto_target(head=pose, duration=0.5)
        self.mini.say("I'm sorry, I don't have that specific information in my current database.")

    def speak(self, text: str):
        """
        Speak text aloud. Intentionally blocking so the _speak_lock in server.py
        (used via asyncio.to_thread) serializes concurrent responses correctly.
        """
        if not self.mini:
            return
        self.mini.say(text)

    # -------------------------------------------------------------------------
    # Idle behaviors — run between conversations
    # -------------------------------------------------------------------------

    @_requires_robot
    def turn_toward_speaker(self, angle_rad: float):
        """Rotate the body to face a direction given by a DoA angle (0 = front, radians)."""
        self.mini.goto_target(body_yaw=angle_rad, duration=0.3)

    @_requires_robot
    def start_idle_behaviors(self):
        """
        Start face tracking and DoA orientation as daemon threads.
        Call stop_idle_behaviors() before any conversation to avoid head movement conflicts.
        """
        self._idle_stop.clear()
        threading.Thread(target=self._face_track_loop, daemon=True).start()
        if self._doa:
            threading.Thread(target=self._doa_loop, daemon=True).start()

    def stop_idle_behaviors(self):
        """Signal idle threads to exit at their next iteration."""
        self._idle_stop.set()

    def _face_track_loop(self):
        """
        Continuously grab camera frames, detect the largest face, and look at it.
        Uses OpenCV's Haar cascade — fast enough for real-time on CPU at _FACE_INTERVAL.
        look_at_image() takes pixel coords (u, v) of the point to look at.
        """
        while not self._idle_stop.is_set():
            try:
                frame = self.mini.media.get_frame()
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = _FACE_DETECTOR.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        self.mini.look_at_image(x + w // 2, y + h // 2, duration=0.3)
            except Exception:
                pass
            time.sleep(_FACE_INTERVAL)

    def _doa_loop(self):
        """
        Poll the ReSpeaker mic array for direction-of-arrival.
        When speech is detected, rotate the body to face the speaker so the
        robot is already oriented before the question is finished.
        """
        while not self._idle_stop.is_set():
            try:
                result = self._doa.get_DoA()
                if result is not None:
                    angle, speech = result
                    if speech:
                        self.turn_toward_speaker(angle)
            except Exception:
                pass
            time.sleep(_DOA_INTERVAL)

    # -------------------------------------------------------------------------
    # Voice input
    # -------------------------------------------------------------------------

    def listen_for_question(self) -> str | None:
        """
        Block until a complete spoken utterance is captured and transcribed.

        Flow:
          1. Open the default microphone via sounddevice at 16kHz mono int16.
          2. Feed 30ms frames into webrtcvad to detect speech onset.
          3. Once speech starts, keep recording until _SILENCE_TIMEOUT seconds
             of consecutive non-speech frames are seen.
          4. Convert raw PCM to float32, run through Whisper, return the transcript.

        Returns the transcribed string, or None if nothing was captured.
        """
        if not self.mini:
            return None

        # aggressiveness 0–3: 2 is a good balance between sensitivity and false positives.
        vad = webrtcvad.Vad(2)
        frames: list[bytes] = []
        speech_started = False
        silence_frames = 0
        silence_threshold = int(_SILENCE_TIMEOUT * 1000 / _VAD_FRAME_MS)

        with sd.RawInputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=_VAD_FRAME_SAMPLES,
        ) as stream:
            while True:
                chunk, _ = stream.read(_VAD_FRAME_SAMPLES)
                is_speech = vad.is_speech(bytes(chunk), _SAMPLE_RATE)

                if is_speech:
                    speech_started = True
                    silence_frames = 0
                    frames.append(bytes(chunk))
                elif speech_started:
                    # Keep buffering silence so we don't cut off trailing words.
                    frames.append(bytes(chunk))
                    silence_frames += 1
                    if silence_frames >= silence_threshold:
                        break

        if not frames:
            return None

        # Whisper expects float32 in [-1.0, 1.0] — divide by int16 max to normalize.
        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = _WHISPER.transcribe(audio, language="en")
        text = " ".join(s.text for s in segments).strip()
        return text or None

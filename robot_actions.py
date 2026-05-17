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

# Reachy Mini head range: pitch ±25°, roll ±20°, yaw ±45°
_ROLL_THINK = 15
_PITCH_CONFUSED = 10

_FACE_INTERVAL = 0.15       # seconds between face detection updates
_DOA_INTERVAL = 0.05        # seconds between DoA polls
_SAMPLE_RATE = 16000        # Hz — required by webrtcvad and Whisper
_VAD_FRAME_MS = 30          # ms — webrtcvad frame size must be 10, 20, or 30
_VAD_FRAME_SAMPLES = _SAMPLE_RATE * _VAD_FRAME_MS // 1000
_SILENCE_TIMEOUT = 1.5      # seconds of silence to end an utterance

_FACE_DETECTOR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_WHISPER = WhisperModel("base", device="cpu", compute_type="int8")


def _requires_robot(fn):
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

        self._idle_stop = threading.Event()
        self._doa = AudioDoA() if self.mini else None

    @_requires_robot
    def thinking(self):
        pose = create_head_pose(roll=_ROLL_THINK, degrees=True)
        self.mini.goto_target(head=pose, duration=0.5)

    @_requires_robot
    def answering(self):
        pose = create_head_pose()
        self.mini.goto_target(head=pose, duration=0.4)

    @_requires_robot
    def greet(self):
        self.mini.wake_up()
        self.mini.say("Hey! I'm Lancer, CBU's ACM chatbot.")

    @_requires_robot
    def confused(self):
        self.mini.antennas.sad()
        pose = create_head_pose(pitch=_PITCH_CONFUSED, degrees=True)
        self.mini.goto_target(head=pose, duration=0.5)

    def speak(self, text: str):
        """Blocking — serialized through _speak_lock in server.py via asyncio.to_thread."""
        if not self.mini:
            return
        self.mini.say(text)

    @_requires_robot
    def turn_toward_speaker(self, angle_rad: float):
        """Rotate body to face a DoA angle (0 = front, radians, +left/-right)."""
        self.mini.goto_target(body_yaw=angle_rad, duration=0.3)

    @_requires_robot
    def start_idle_behaviors(self):
        """Launch face tracking and DoA orientation background threads."""
        self._idle_stop.clear()
        threading.Thread(target=self._face_track_loop, daemon=True).start()
        if self._doa:
            threading.Thread(target=self._doa_loop, daemon=True).start()

    def stop_idle_behaviors(self):
        self._idle_stop.set()

    def _face_track_loop(self):
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

    def listen_for_question(self) -> str | None:
        """
        Block until a complete spoken utterance is captured and transcribed.
        Uses webrtcvad for onset/offset detection, faster-whisper for STT.
        Returns transcribed text or None if nothing was captured.
        """
        if not self.mini:
            return None

        vad = webrtcvad.Vad(2)  # aggressiveness 0–3; 2 balances sensitivity vs. false positives
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
                    frames.append(bytes(chunk))
                    silence_frames += 1
                    if silence_frames >= silence_threshold:
                        break

        if not frames:
            return None

        audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = _WHISPER.transcribe(audio, language="en")
        text = " ".join(s.text for s in segments).strip()
        return text or None

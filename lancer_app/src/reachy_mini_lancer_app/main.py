"""
Lancer -- CBU's RAG-powered campus robot.

Runs on the Reachy Mini and follows the same lifecycle as the stock apps: a
ReachyMiniApp subclass whose run() is driven by the daemon and exits when
stop_event is set.

The robot has no TTS engine and limited RAM, so it only captures and plays
audio. Speech-to-text, retrieval, generation and speech synthesis all happen on
the RAG server (see server.py), reached over the wired link.

    idle      -- native head tracking follows whoever is in frame
    listening -- energy-gated capture from the ReSpeaker array
    thinking  -- head tilts while the server runs RAG
    answering -- head returns to neutral, answer plays through the speaker
    confused  -- antennas droop and head dips when retrieval found nothing
"""

import io
import logging
import math
import os
import threading
import time
import wave

import numpy as np
import requests
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)

# --- Config -----------------------------------------------------------------
# The Mac hosting Chroma + Ollama, reachable over the USB-Ethernet link.
# The wired link is fastest but is not always plugged in; Tailscale always
# works. Try each in order at startup and use whichever answers.
SERVER_CANDIDATES = [
    os.environ.get("LANCER_SERVER"),
    "http://10.5.1.82:7870",        # USB-Ethernet link, ~14 ms
    "http://100.87.191.40:7870",    # Tailscale, ~50 ms
]
SERVER_URL = SERVER_CANDIDATES[1]


def pick_server() -> str:
    """Return the first reachable RAG server, preferring the wired link."""
    global SERVER_URL
    for url in SERVER_CANDIDATES:
        if not url:
            continue
        try:
            if requests.get(f"{url}/health", timeout=4).ok:
                SERVER_URL = url
                logger.info("Using RAG server at %s", url)
                return url
        except Exception:
            continue
    logger.warning("No RAG server reachable; defaulting to %s", SERVER_URL)
    return SERVER_URL
REQUEST_TIMEOUT = 120        # seconds; local LLM generation can be slow

SAMPLE_RATE = 16000          # matches reachy_mini.media AudioBase.SAMPLE_RATE

# --- Voice activity detection ---
# The ReSpeaker applies its own gain and noise suppression, so a simple RMS
# gate is enough here -- webrtcvad would mean adding a compiled dependency.
_NOISE_SETTLE = 1.5          # seconds to let wake-up motor noise die down first
_NOISE_CALIBRATION = 2.0     # seconds of ambient sampled to set the gate
_SPEECH_MULTIPLIER = 2.0     # speech must exceed noise floor by this factor
_SPEECH_RMS_FLOOR = 0.025    # never gate below this, however quiet the room
_SPEECH_RMS_CEILING = 0.045  # never gate above this, or quiet speech is missed
_SILENCE_TIMEOUT = 0.7       # seconds of silence that end an utterance
_MIN_UTTERANCE = 0.6         # ignore anything shorter (coughs, door clicks)
_MIN_VOICED = 0.45           # seconds of actually-loud audio required
_MAX_UTTERANCE = 15.0        # hard cap so a noisy room can't record forever

# --- Barge-in ---
# The board runs software AEC, so the mic feed has most of Lancer's own voice
# removed -- but not all of it, so the barge-in gate sits above the speech gate.
_BARGE_MULTIPLIER = 1.6      # barge-in floor relative to the speech gate
_BARGE_CHUNKS = 3            # consecutive loud reads before we accept an interrupt
_BARGE_GRACE = 0.8           # seconds of playback before barge-in can trigger
_POST_SPEECH_DRAIN = 0.6     # seconds of mic input discarded after Lancer speaks
_BARGE_ECHO_HEADROOM = 1.4   # interrupt must exceed measured echo by this factor

# --- Direction of arrival ---
_DOA_INTERVAL = 0.05         # seconds between DoA polls (ReSpeaker updates ~20Hz)
_DOA_DEADBAND = 0.45         # radians (~26 deg) of change before the body moves
_DOA_MIN_INTERVAL = 1.5      # seconds between body turns, so it cannot jitter
_DOA_SUSTAIN = 2             # consecutive speech reads before turning
_FACE_STICKY = 6.0           # seconds a face still counts as present after loss

# --- Hybrid vision ---
# YOLO on the Mac sees whole people, so it holds on to someone who is too close,
# turned away, or off to the side -- all cases where the on-board face detector
# gives up. YuNet stays as the fallback when the Mac is unreachable.
_TRACK_INTERVAL = 0.12       # seconds between frames sent for detection
_TRACK_FALLBACK_AFTER = 2.0  # seconds without a YOLO hit before handing back
_TRACK_AIM_DURATION = 0.5    # seconds for each look_at_image move
# YOLO re-detects every frame and the box drifts a few pixels each time. Aiming
# at the raw centre makes the head chase that noise, so smooth it and ignore
# movement too small to be a real person moving.
_AIM_SMOOTHING = 0.45        # EMA weight for each new detection
_AIM_DEADBAND_FRAC = 0.022   # of frame width -- below this, do not move
_AIM_MIN_INTERVAL = 0.18     # seconds between commanded moves
_AIM_RESET_AFTER = 1.5       # seconds of no detection before the filter resets
_DOA_MAX_YAW = 2.0           # radians (~115 deg); beyond this the IK refuses
_DOA_FACE_GRACE = 2.5        # seconds without a face before sound may steer us

# --- Poses ---
_ANTENNA_DROOP = 0.4         # radians, symmetric droop for the confused pose
_ANTENNA_PERK = 0.3          # radians, alert tilt while thinking
# Vertical (0.0) is an unstable equilibrium for the antenna gearboxes -- they
# hunt around it and visibly shake. Pollen's guidance is to bias a few degrees
# off vertical so gravity takes up the backlash.
_ANTENNA_NEUTRAL = 0.18      # radians, ~10 degrees

GREETING = "Hey! I'm Lancer, CBU's ACM robot. Ask me about the university."


class Lancer:
    """Robot behaviour for one Lancer session."""

    def __init__(self, mini: ReachyMini, cfg: dict | None = None):
        self.mini = mini
        self.cfg = cfg or {}
        self.speech_rms = _SPEECH_RMS_FLOOR
        self.speaking = False
        self._last_yaw = 0.0
        self._last_turn = 0.0
        self._last_face = 0.0
        self._last_person = 0.0
        self.vision_mode = "yunet"
        self.request_aim_reset = False

    @property
    def barge_rms(self) -> float:
        return self.speech_rms * _BARGE_MULTIPLIER

    def opt(self, key: str, default):
        value = self.cfg.get(key, default)
        return default if value is None else value

    # -- capture setup -------------------------------------------------------

    @staticmethod
    def _mono(chunk: np.ndarray) -> np.ndarray:
        """The ReSpeaker returns 2 channels; average them to mono."""
        return chunk.mean(axis=1) if chunk.ndim == 2 else chunk

    def calibrate(self) -> None:
        """
        Measure the ambient noise floor and set the speech gate above it.

        The room's baseline varies a lot between a quiet office and a hallway,
        and a fixed threshold either misses quiet speech or triggers on HVAC.
        """
        # Motors settle after wake_up(); sampling too early reads their noise
        # as the room's floor and leaves the gate far too high.
        time.sleep(_NOISE_SETTLE)

        levels: list[float] = []
        deadline = time.monotonic() + _NOISE_CALIBRATION
        while time.monotonic() < deadline:
            chunk = self.mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.01)
                continue
            mono = self._mono(chunk)
            levels.append(float(np.sqrt(np.mean(np.square(mono)))))

        if levels:
            # 25th percentile tracks the true floor even if someone talks or a
            # door slams during calibration; median would be dragged upward.
            noise = float(np.percentile(levels, 25))
            self.speech_rms = min(
                max(noise * float(self.opt("speech_multiplier", _SPEECH_MULTIPLIER)),
                    _SPEECH_RMS_FLOOR),
                float(self.opt("speech_rms_ceiling", _SPEECH_RMS_CEILING)),
            )
        logger.info(
            "Noise floor calibrated: gate=%.4f barge=%.4f (%d samples, p25=%.4f)",
            self.speech_rms, self.barge_rms, len(levels),
            float(np.percentile(levels, 25)) if levels else -1.0,
        )

    # -- gestures ------------------------------------------------------------

    # Expression is antenna-only: head poses fight the daemon's visual tracking,
    # and staying locked on the person matters more than a tilt gesture.

    def thinking(self) -> None:
        self.mini.set_target_antenna_joint_positions([_ANTENNA_PERK, -_ANTENNA_PERK])

    def answering(self) -> None:
        # Mirrored so the pair looks symmetric while both stay off vertical.
        self.mini.set_target_antenna_joint_positions(
            [_ANTENNA_NEUTRAL, -_ANTENNA_NEUTRAL]
        )

    def confused(self) -> None:
        self.mini.set_target_antenna_joint_positions([-_ANTENNA_DROOP, _ANTENNA_DROOP])

    # -- audio ---------------------------------------------------------------

    def listen(self, stop_event: threading.Event) -> np.ndarray | None:
        """
        Capture one utterance from the mic array.

        Returns float32 samples at SAMPLE_RATE, or None if the app was asked to
        stop or nothing worth sending was captured.
        """
        collected: list[np.ndarray] = []
        voiced_samples = 0
        speaking = False
        silence_started: float | None = None
        began = time.monotonic()

        while not stop_event.is_set():
            chunk = self.mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.01)
                continue

            mono = self._mono(chunk)
            rms = float(np.sqrt(np.mean(np.square(mono))))
            if rms >= self.speech_rms:
                speaking = True
                silence_started = None
                voiced_samples += len(mono)
                collected.append(mono)
            elif speaking:
                # Keep trailing silence so final words aren't clipped.
                collected.append(mono)
                silence_started = silence_started or time.monotonic()
                if time.monotonic() - silence_started >= _SILENCE_TIMEOUT:
                    break

            if speaking and time.monotonic() - began > _MAX_UTTERANCE:
                break

        if not collected:
            return None
        audio = np.concatenate(collected)
        if len(audio) / SAMPLE_RATE < _MIN_UTTERANCE:
            return None
        if voiced_samples / SAMPLE_RATE < _MIN_VOICED:
            # Mostly silence around a brief spike -- a door, a cough, a chair.
            return None
        return audio

    def play(self, audio: np.ndarray, stop_event: threading.Event) -> bool:
        """
        Play float32 samples, listening for the user talking over the top.

        Everything is pushed into the playback queue at once and the mic is
        watched for the expected duration; on a genuine interrupt the queue is
        dropped by restarting the output pipeline so Lancer stops mid-sentence.

        Returns True if playback was interrupted by the user.
        """
        self.speaking = True
        try:
            # The output pipeline is 2-channel; pushing mono is read as
            # interleaved stereo and plays at double speed.
            stereo = np.column_stack([audio, audio]).astype(np.float32)
            self.mini.media.push_audio_sample(stereo)
            started = time.monotonic()
            deadline = started + len(audio) / SAMPLE_RATE + 0.3
            grace_until = started + float(self.opt("barge_grace", _BARGE_GRACE))
            barge_enabled = bool(self.opt("barge_enabled", True))
            echo: list[float] = []
            threshold = self.barge_rms
            peak = 0.0
            loud = 0
            while time.monotonic() < deadline and not stop_event.is_set():
                chunk = self.mini.media.get_audio_sample()
                if chunk is None or len(chunk) == 0:
                    time.sleep(0.01)
                    continue
                mono = self._mono(chunk)
                rms = float(np.sqrt(np.mean(np.square(mono))))

                # During the grace window Lancer is the only thing talking, so
                # whatever the mic hears is AEC residual. Calibrate against it.
                if time.monotonic() < grace_until:
                    echo.append(rms)
                    continue
                if echo:
                    measured = float(np.percentile(echo, 90))
                    threshold = max(
                        self.barge_rms,
                        measured * float(self.opt("barge_echo_headroom",
                                                  _BARGE_ECHO_HEADROOM)),
                    )
                    logger.info(
                        "Barge-in armed: echo_p90=%.4f threshold=%.4f "
                        "(speak over Lancer louder than this to interrupt)",
                        measured, threshold,
                    )
                    peak = 0.0
                    logger.debug("Barge-in threshold for this reply: %.4f", threshold)
                    echo = []

                if not barge_enabled:
                    continue
                peak = max(peak, rms)
                loud = loud + 1 if rms >= threshold else 0
                if loud >= _BARGE_CHUNKS:
                    # MediaManager has no clear_player(); dropping and restarting
                    # the output pipeline is what cuts queued audio immediately.
                    self.mini.media.stop_playing()
                    self.mini.media.start_playing()
                    logger.info("Barge-in detected -- stopping playback")
                    return True
            logger.info("Playback finished; loudest mic reading was %.4f "
                        "(threshold %.4f)", peak, threshold)
            return False
        finally:
            self.drain_microphone()
            self.speaking = False

    def drain_microphone(self) -> None:
        """
        Discard buffered mic audio after playback.

        get_audio_sample() returns queued chunks, so whatever Lancer just said is
        still sitting in the buffer when listening resumes -- it would transcribe
        its own answer and reply to itself.
        """
        deadline = time.monotonic() + _POST_SPEECH_DRAIN
        dropped = 0
        while time.monotonic() < deadline:
            chunk = self.mini.media.get_audio_sample()
            if chunk is None or len(chunk) == 0:
                time.sleep(0.01)
                continue
            dropped += len(chunk)
        if dropped:
            logger.debug("Drained %.2fs of post-speech audio", dropped / SAMPLE_RATE)

    # -- direction of arrival ------------------------------------------------

    def person_recently_seen(self) -> bool:
        """True if YOLO saw a person recently -- used by the addressing gate."""
        return (time.monotonic() - self._last_person) <= _FACE_STICKY

    def vision_loop(self, stop_event: threading.Event) -> None:
        """
        Drive the head from whole-body detection, falling back to YuNet.

        While YOLO is answering we own the head via look_at_image; if the Mac
        stops responding or nobody is in frame we hand the head back to the
        daemon's own face tracking rather than freezing.
        """
        if not bool(self.opt("yolo_tracking", True)):
            logger.info("YOLO tracking disabled by config")
            return

        session = requests.Session()
        aim_u = aim_v = None          # smoothed target
        sent_u = sent_v = None        # last position actually commanded
        last_move = 0.0
        while not stop_event.is_set():
            time.sleep(_TRACK_INTERVAL)
            if self.speaking:
                continue
            try:
                jpeg = self.mini.media.get_frame_jpeg()
                if not jpeg:
                    continue
                reply = session.post(
                    f"{SERVER_URL}/api/track",
                    files={"file": ("f.jpg", jpeg, "image/jpeg")},
                    timeout=4,
                ).json()
            except Exception:
                reply = None

            now = time.monotonic()
            if reply and reply.get("found"):
                if now - self._last_person > _AIM_RESET_AFTER or self.request_aim_reset:
                    aim_u = aim_v = sent_u = sent_v = None
                    self.request_aim_reset = False
                self._last_person = now
                if self.vision_mode != "yolo":
                    # Take the head off the daemon tracker before aiming it.
                    self.mini.stop_head_tracking()
                    self.vision_mode = "yolo"
                    logger.info("Vision: YOLO (whole body)")

                u, v = float(reply["u"]), float(reply["v"])
                if aim_u is None:
                    aim_u, aim_v = u, v
                else:
                    aim_u += _AIM_SMOOTHING * (u - aim_u)
                    aim_v += _AIM_SMOOTHING * (v - aim_v)

                deadband = _AIM_DEADBAND_FRAC * float(reply.get("width") or 640)
                moved_enough = (
                    sent_u is None
                    or math.hypot(aim_u - sent_u, aim_v - sent_v) >= deadband
                )
                if moved_enough and now - last_move >= _AIM_MIN_INTERVAL:
                    try:
                        self.mini.look_at_image(
                            int(aim_u), int(aim_v), duration=_TRACK_AIM_DURATION
                        )
                        sent_u, sent_v = aim_u, aim_v
                        last_move = now
                    except Exception:
                        pass
            elif now - self._last_person >= _TRACK_FALLBACK_AFTER:
                if self.vision_mode != "yunet":
                    self.mini.start_head_tracking(
                        weight=float(self.opt("head_tracking_weight", 1.0))
                    )
                    self.vision_mode = "yunet"
                    logger.info("Vision: YuNet fallback (no person / server down)")

    def face_recently_seen(self) -> bool:
        """True if a face is visible now or was within _FACE_STICKY seconds."""
        if self._face_visible():
            self._last_face = time.monotonic()
            return True
        if self.person_recently_seen():
            return True
        return (time.monotonic() - self._last_face) <= _FACE_STICKY

    def _face_visible(self) -> bool:
        """True when the camera currently has a face; DoA defers to vision."""
        try:
            target = self.mini.get_tracked_face(wait=False)
            return bool(getattr(target, "detected", False))
        except Exception:
            return False

    @staticmethod
    def _normalize(angle: float) -> float:
        """Wrap a DoA bearing into [-pi, pi]; raw values overrun the yaw limit."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def doa_loop(self, stop_event: threading.Event) -> None:
        """
        Turn the body toward whoever is speaking.

        Head tracking handles the face once it is in frame; DoA is what gets
        Lancer facing the right way when someone speaks from off-camera.
        """
        if not bool(self.opt("doa_enabled", True)):
            return
        sustained = 0
        while not stop_event.is_set():
            try:
                result = self.mini.media.get_DoA()
                if result is not None:
                    raw, is_speech = result
                    angle = max(-_DOA_MAX_YAW, min(_DOA_MAX_YAW, self._normalize(raw)))
                    now = time.monotonic()
                    # Ignore DoA while speaking (AEC residual steers it), require
                    # a real change, and rate-limit so the body cannot jitter.
                    face = self._face_visible()
                    if face:
                        self._last_face = now
                    # With doa_priority the speaker wins even if a face is in
                    # frame -- Lancer should face whoever is talking to it.
                    if face and not bool(self.opt("doa_priority", True)):
                        continue

                    # A single frame of speech is a cough or a door. Require the
                    # bearing to hold before committing the body to a turn.
                    sustained = sustained + 1 if is_speech else 0

                    if (
                        sustained >= _DOA_SUSTAIN
                        and not self.speaking
                        and (bool(self.opt("doa_priority", True))
                             or now - self._last_face >= _DOA_FACE_GRACE)
                        and abs(angle - self._last_yaw) >= _DOA_DEADBAND
                        and now - self._last_turn >= _DOA_MIN_INTERVAL
                    ):
                        # Hand the body back to us briefly, turn, then return it
                        # to vision -- two writers at once makes the IK fail.
                        try:
                            self.mini.set_automatic_body_yaw(False)
                            self.mini.set_target_body_yaw(angle)
                            # Turning the body alone leaves the head pointing
                            # wherever it last tracked someone. Centre it so the
                            # camera actually looks where the sound came from.
                            self.mini.goto_target(
                                head=create_head_pose(), duration=0.4
                            )
                            self.request_aim_reset = True
                        finally:
                            self.mini.set_automatic_body_yaw(True)
                        logger.info("Turned toward speech at %.2f rad", angle)
                        self._last_yaw = angle
                        self._last_turn = now
                        sustained = 0
            except Exception:
                pass
            time.sleep(_DOA_INTERVAL)


def to_wav_bytes(audio: np.ndarray) -> bytes:
    """Encode float32 samples as 16-bit mono WAV for upload."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_bytes_to_float32(data: bytes) -> np.ndarray:
    """Decode a 16-bit mono WAV payload back to float32 samples."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def fetch_config() -> dict:
    """
    Pull behaviour settings from the dashboard's runtime config.

    Keeping one source of truth means the dashboard can retune the robot without
    editing code here; on any failure we fall back to the module defaults.
    """
    try:
        r = requests.get(f"{SERVER_URL}/api/config", timeout=8)
        cfg = r.json().get("config", {})
        logger.info("Loaded runtime config from server")
        return cfg
    except Exception as e:
        logger.warning("Using built-in defaults (config fetch failed: %s)", e)
        return {}


def ask_server(audio: np.ndarray, face_present: bool = False) -> dict | None:
    """
    POST one utterance to the RAG server and return its JSON reply.

    face_present tells the server whether anyone is actually looking at Lancer,
    which it uses to decide if the speech was aimed at the robot at all.
    """
    import base64

    try:
        response = requests.post(
            f"{SERVER_URL}/voice",
            params={"face": str(bool(face_present)).lower()},
            files={"file": ("utterance.wav", to_wav_bytes(audio), "audio/wav")},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as e:
        logger.error("Lancer server unreachable at %s: %s", SERVER_URL, e)
        return None

    if response.status_code in (204, 422):
        # 422 = nothing intelligible; 204 = overheard, not addressed to Lancer.
        return None
    if not response.ok:
        logger.error("Lancer server error %s: %s", response.status_code, response.text[:200])
        return None

    payload = response.json()
    payload["audio"] = wav_bytes_to_float32(base64.b64decode(payload["audio_b64"]))
    return payload


class LancerApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for Lancer."""

    dont_start_webserver = True

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        pick_server()
        cfg = fetch_config()
        lancer = Lancer(reachy_mini, cfg)

        # wake_up() plays the pose and sound but does NOT enable torque -- without
        # this the head flops back into the shell and every command still reports
        # success. Motors first, then wake.
        reachy_mini.enable_motors()
        reachy_mini.wake_up()
        # get_audio_sample() returns None until the audio device is recording.
        reachy_mini.acquire_media()
        reachy_mini.media.start_recording()
        # push_audio_sample() needs the output pipeline running first.
        try:
            reachy_mini.media.start_playing()
        except Exception as e:
            logger.warning("start_playing() failed (may auto-start): %s", e)
        lancer.calibrate()
        # Head follows the face; the body follows the head, so Lancer turns to
        # face people rather than craning at them. Tracking stays on throughout.
        # start_head_tracking() only enables the aiming loop -- the face detector
        # itself is separate, and without it the tracker never sees anyone and the
        # head just recenters on the lost-target timeout.
        try:
            requests.post(f"http://127.0.0.1:8000/api/media/tracking/enable",
                          json={}, timeout=8)
            logger.info("Face detector enabled")
        except Exception as e:
            logger.warning("Could not enable face detector: %s", e)
        reachy_mini.start_head_tracking(
            weight=float(cfg.get("head_tracking_weight") or 1.0)
        )
        reachy_mini.set_automatic_body_yaw(True)

        if not bool(cfg.get("doa_enabled", True)):
            logger.info("Audio direction tracking disabled by config")
        doa_thread = threading.Thread(
            target=lancer.doa_loop, args=(stop_event,), daemon=True
        )
        doa_thread.start()
        vision_thread = threading.Thread(
            target=lancer.vision_loop, args=(stop_event,), daemon=True
        )
        vision_thread.start()
        logger.info("Audio direction tracking: on (deadband %.2f rad)", _DOA_DEADBAND)
        logger.info("Lancer ready -- server at %s", SERVER_URL)

        try:
            while not stop_event.is_set():
                audio = lancer.listen(stop_event)
                if audio is None:
                    continue

                logger.info("Captured %.1fs of speech", len(audio) / SAMPLE_RATE)

                # Ask first, and only disturb the pose once there is something to
                # say. Gesturing on every noise trigger is what made Lancer twitch.
                # A face that was visible moments ago still counts: detection
                # drops out constantly when someone turns their head to speak.
                face_present = lancer.face_recently_seen()
                reply = ask_server(audio, face_present)
                if reply is None:
                    # Overheard chatter: say nothing, move nothing.
                    continue
                lancer.thinking()
                logger.info("heard=%r answer=%r", reply["heard"], reply["answer"])

                # Keep following the user while replying -- only the antennas move.
                if reply["unknown"]:
                    lancer.confused()
                else:
                    lancer.answering()
                try:
                    lancer.play(reply["audio"], stop_event)
                finally:
                    lancer.answering()
        finally:
            try:
                reachy_mini.set_automatic_body_yaw(False)
            except Exception:
                pass
            reachy_mini.stop_head_tracking()
            for cleanup in (
                reachy_mini.media.stop_recording,
                reachy_mini.goto_sleep,
                reachy_mini.disable_motors,
            ):
                try:
                    cleanup()
                except Exception:
                    pass


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = LancerApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""
Audition Piper TTS voices for Lancer.

Downloads voice models on demand into ./voices, synthesizes the same line with
each, and plays them back so they can be compared directly.

    python audition_voices.py --list
    python audition_voices.py --voices en_US-ryan-high en_US-joe-medium
    python audition_voices.py --male            # curated male shortlist
    python audition_voices.py --male --text "What programs does CBU offer?"

Browse pre-rendered samples without downloading anything:
    https://rhasspy.github.io/piper-samples/
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

VOICES_DIR = Path(__file__).parent / "voices"
PIPER = Path(__file__).parent / ".venv" / "bin" / "piper"
INDEX_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

DEFAULT_LINE = "Hey, I'm Lancer, CBU's ACM robot. Ask me about the university."

# Voices that read as male in en_US/en_GB -- a reasonable starting shortlist.
MALE_SHORTLIST = [
    "en_US-ryan-high",
    "en_US-joe-medium",
    "en_US-john-medium",
    "en_US-hfc_male-medium",
    "en_US-bryce-medium",
    "en_US-norman-medium",
    "en_GB-alan-medium",
    "en_GB-northern_english_male-medium",
]


def load_index() -> dict:
    cache = VOICES_DIR / "voices.json"
    if not cache.exists():
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(INDEX_URL, cache)
    return json.loads(cache.read_text())


def list_voices(index: dict) -> None:
    rows = []
    for key, v in index.items():
        code = v.get("language", {}).get("code", "")
        if code.startswith("en_"):
            rows.append((code, v["name"], v["quality"], key))
    for code, name, quality, key in sorted(rows):
        print(f"  {key:<42} {code} {name} ({quality})")
    print(f"\n  {len(rows)} English voices. Pass any key above to --voices.")


def ensure_model(key: str, index: dict) -> Path | None:
    """Download the .onnx and .onnx.json for a voice key if not already present."""
    entry = index.get(key)
    if entry is None:
        print(f"  ! unknown voice: {key}")
        return None

    # The index lists real repo paths -- use them rather than guessing a layout.
    onnx_rel = next((f for f in entry["files"] if f.endswith(".onnx")), None)
    json_rel = next((f for f in entry["files"] if f.endswith(".onnx.json")), None)
    if not onnx_rel or not json_rel:
        print(f"  ! no model files listed for {key}")
        return None

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    dest = VOICES_DIR / f"{key}.onnx"
    if not dest.exists():
        print(f"  downloading {key} ...", flush=True)
        urllib.request.urlretrieve(f"{BASE_URL}/{onnx_rel}", dest)
        urllib.request.urlretrieve(f"{BASE_URL}/{json_rel}", VOICES_DIR / f"{key}.onnx.json")
    return dest


def announce(text: str) -> None:
    """Speak a label with the system voice so samples are distinguishable."""
    subprocess.run(["say", "-v", "Samantha", "-r", "200", text],
                   check=False, capture_output=True)


def play(model: Path, line: str, key: str) -> None:
    wav = VOICES_DIR / f"_sample_{key}.wav"
    proc = subprocess.run(
        [str(PIPER), "-m", str(model), "-f", str(wav)],
        input=line.encode(), capture_output=True,
    )
    if proc.returncode != 0 or not wav.exists() or wav.stat().st_size == 0:
        print(f"  ! synthesis failed for {key}: {proc.stderr.decode()[:120]}")
        return
    subprocess.run(["afplay", str(wav)], check=False)
    print(f"  played {key}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="List all English voices")
    ap.add_argument("--voices", nargs="+", help="Voice keys to audition")
    ap.add_argument("--male", action="store_true", help="Audition the male shortlist")
    ap.add_argument("--text", default=DEFAULT_LINE, help="Line to speak")
    args = ap.parse_args()

    index = load_index()
    if args.list:
        list_voices(index)
        return 0

    keys = args.voices or (MALE_SHORTLIST if args.male else None)
    if not keys:
        ap.print_help()
        return 1

    for key in keys:
        model = ensure_model(key, index)
        if model is None:
            continue
        announce(key.replace("en_US-", "").replace("en_GB-", "").replace("-", " "))
        play(model, args.text, key)
    return 0


if __name__ == "__main__":
    sys.exit(main())

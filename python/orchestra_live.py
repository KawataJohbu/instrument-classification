from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from pprint import pformat

import sounddevice as sd

import orchestra

# コンソール表示
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_WINDOW_SECONDS = 3.0
DEFAULT_MIN_RMS = 150.0
MODEL_CACHE = Path(__file__).resolve().parent / ".cache" / "orchestra_live.joblib"


def load_or_train_model(data_dir: Path, max_seconds: float | None,
                        segment_seconds: float, retrain: bool = False):
    return orchestra.load_or_train_shared_model(
        data_dir, max_seconds, segment_seconds, retrain=retrain, cache_path=MODEL_CACHE,
    )


def record_mono_samples(sample_rate: int, seconds: float, device: int | None) -> list[float]:
    frame_count = int(sample_rate * seconds)
    recording = sd.rec(
        frame_count,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()

    # Scale microphone input to the same rough range as 16-bit wav samples.
    return [float(value[0] * 32768.0) for value in recording]


def print_ranked_scores(scores: dict[str, float], threshold: float) -> None:
    ranked_scores = sorted(scores.items(), key=lambda row: row[1], reverse=True)
    detected = [(label, score) for label, score in ranked_scores if score >= threshold]

    print("Relative instrument scores")
    for label, score in ranked_scores:
        marker = "*" if score >= threshold else " "
        line = f"{marker} {label:7}: {score:6.1f} / 100"
        if score >= threshold:
            line = f"\033[32m{line}\033[0m"
        print(line)


def safe_print(text: object) -> None:
    message = str(text)
    print(message.encode("cp932", errors="replace").decode("cp932"))


def add_input_options(parser):
    parser.add_argument("--loopback", action="store_true", help="Capture Windows playback instead of the microphone.")
    parser.add_argument("--loopback-device", type=int, help="PyAudioWPatch loopback index (different from --device).")
    parser.add_argument("--list-loopback-devices", action="store_true", help="List Windows playback loopback devices and exit.")
    parser.add_argument("--timing", action="store_true", help="Print capture and analysis durations.")


def print_loopback_devices():
    import audio_loopback
    for info in audio_loopback.list_devices():
        safe_print(f"{'>' if info['selected'] else ' '} {info['index']}: {info['name']} ({int(info['defaultSampleRate'])} Hz)")


def record_input(args, stop=None):
    if args.loopback:
        import audio_loopback
        return audio_loopback.record(args.window_seconds, args.loopback_device, stop)
    return record_mono_samples(args.sample_rate, args.window_seconds, args.device), args.sample_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify live microphone audio as violin/cello/flute/oboe/horn."
    )
    add_input_options(parser)
    parser.add_argument(
        "--retrain", action="store_true",
        help="Rebuild and save the model even if a saved model is available.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(orchestra.DATA_DIR),
        help="Training data root. Expect subfolders such as violin/cello/flute/oboe/horn.",
    )
    parser.add_argument(
        "--training-max-seconds",
        type=float,
        default=orchestra.DEFAULT_MAX_SECONDS,
        help="Seconds analyzed from each training wav. Use 0 to analyze full files.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=orchestra.DEFAULT_SEGMENT_SECONDS,
        help="Seconds per training segment.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Seconds recorded for each live prediction.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Microphone recording sample rate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=orchestra.DEFAULT_PRESENCE_THRESHOLD,
        help="Score threshold for showing detected instruments.",
    )
    parser.add_argument(
        "--min-rms",
        type=float,
        default=DEFAULT_MIN_RMS,
        help="Skip prediction when microphone volume is below this RMS value.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input device number.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one live prediction and exit.",
    )
    args = parser.parse_args()
    if args.loopback_device is not None and not args.loopback:
        parser.error("--loopback-device requires --loopback")
    if args.loopback and args.device is not None:
        parser.error("Use --loopback-device for loopback, not --device")
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0 or args.sample_rate <= 0:
        parser.error("Recording duration and sample rate must be positive")
    return args


def main() -> None:
    args = parse_args()

    if args.list_loopback_devices:
        print_loopback_devices()
        return

    if args.list_devices:
        safe_print(pformat(sd.query_devices()))
        return

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (orchestra.PROJECT_ROOT / data_dir).resolve()

    max_seconds = orchestra.normalize_max_seconds(args.training_max_seconds)
    model = load_or_train_model(
        data_dir, max_seconds, args.segment_seconds, retrain=args.retrain,
    )
    print("Ready. Press Ctrl+C to stop.")
    print()

    try:
        while True:
            print(f"Recording {args.window_seconds:.1f}s...")
            started = time.perf_counter()
            samples, sample_rate = record_input(args)
            recorded = time.perf_counter()
            rms = orchestra.calculate_rms(samples)
            # print(f"RMS: {rms:.1f}")

            if rms < args.min_rms:
                print(f"{RED}スキップ{RESET}")
                time.sleep(0.2)
                if args.once:
                    return
                continue

            features = orchestra.extract_features_from_samples(
                Path("live_microphone.wav"),
                samples,
                sample_rate,
            )
            scores, votes = orchestra.sklearn_file_scores(model, [features])
            if args.timing:
                print(f"Capture: {recorded-started:.3f}s; analysis: {time.perf_counter()-recorded:.3f}s")
            predicted = max(scores, key=scores.get)

            print(f"Top prediction: {predicted}  votes={votes.get(predicted, 0)}")
            print_ranked_scores(scores, threshold=args.threshold)
            print()
            if args.once:
                return
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

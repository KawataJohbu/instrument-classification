from __future__ import annotations

import argparse
import time
from pathlib import Path
from pprint import pformat

import sounddevice as sd

import orchestra


DEFAULT_SAMPLE_RATE = 44100
DEFAULT_WINDOW_SECONDS = 3.0
DEFAULT_MIN_RMS = 300.0


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
        print(f"{marker} {label:7}: {score:6.1f} / 100")

    if detected:
        print("Detected: " + ", ".join(label for label, _score in detected))
    else:
        print(f"Detected: none over threshold. Top candidate: {ranked_scores[0][0]}")


def safe_print(text: object) -> None:
    message = str(text)
    print(message.encode("cp932", errors="replace").decode("cp932"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify live microphone audio as violin/cello/flute/oboe/horn."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        safe_print(pformat(sd.query_devices()))
        return

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (orchestra.PROJECT_ROOT / data_dir).resolve()

    max_seconds = orchestra.normalize_max_seconds(args.training_max_seconds)
    print("Loading training data...")
    train_items = orchestra.load_training_items(
        data_dir,
        max_seconds=max_seconds,
        segment_seconds=args.segment_seconds,
    )
    print(f"Training segments: {len(train_items)}")
    print("Training sklearn model...")
    model = orchestra.train_sklearn_model(train_items)
    print("Ready. Press Ctrl+C to stop.")
    print()

    try:
        while True:
            print(f"Recording {args.window_seconds:.1f}s...")
            samples = record_mono_samples(
                sample_rate=args.sample_rate,
                seconds=args.window_seconds,
                device=args.device,
            )
            rms = orchestra.calculate_rms(samples)
            print(f"RMS: {rms:.1f}")

            if rms < args.min_rms:
                print("Skipped: input is too quiet.")
                print()
                time.sleep(0.2)
                continue

            features = orchestra.extract_features_from_samples(
                Path("live_microphone.wav"),
                samples,
                args.sample_rate,
            )
            scores, votes = orchestra.sklearn_file_scores(model, [features])
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

from __future__ import annotations

import argparse
import math
import statistics
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "wavlist" / "datas_wav"
WINDOW_MS = 50


@dataclass
class SampleFeatures:
    filename: str
    label: str
    note: str
    duration_s: float
    attack_time_s: float
    sustain_ratio: float
    tail_ratio: float
    peak_mean_ratio: float
    centroid_ratio: float

    def vector(self) -> list[float]:
        return [
            self.attack_time_s,
            self.sustain_ratio,
            self.tail_ratio,
            self.peak_mean_ratio,
            self.centroid_ratio,
        ]


def parse_label_and_note(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.startswith("piano"):
        return "piano", stem.removeprefix("piano")
    if stem.startswith("organ"):
        return "organ", stem.removeprefix("organ")
    return "unknown", stem


def load_rms_windows(path: Path, window_ms: int = WINDOW_MS) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())

    samples_per_window = max(1, int(sample_rate * window_ms / 1000))
    mono_samples = decode_mono_samples(raw, sample_width, channels)
    rms_values: list[float] = []

    for start in range(0, len(mono_samples) - samples_per_window + 1, samples_per_window):
        chunk = mono_samples[start:start + samples_per_window]
        rms_values.append(calculate_rms(chunk))

    return rms_values, sample_rate


def decode_mono_samples(raw: bytes, sample_width: int, channels: int) -> list[float]:
    if sample_width == 1:
        values = [sample - 128 for sample in raw]
    elif sample_width == 2:
        count = len(raw) // 2
        values = list(struct.unpack(f"<{count}h", raw))
    elif sample_width == 4:
        count = len(raw) // 4
        values = list(struct.unpack(f"<{count}i", raw))
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels == 1:
        return [float(value) for value in values]

    mono_samples: list[float] = []
    for index in range(0, len(values), channels):
        frame = values[index:index + channels]
        mono_samples.append(float(sum(frame) / len(frame)))
    return mono_samples


def calculate_rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    return math.sqrt(mean_square)


def trim_silence(rms_values: list[float]) -> list[float]:
    if not rms_values:
        return []

    peak = max(rms_values)
    threshold = max(30.0, peak * 0.10)

    start = 0
    while start < len(rms_values) and rms_values[start] < threshold:
        start += 1

    end = len(rms_values) - 1
    while end >= 0 and rms_values[end] < threshold:
        end -= 1

    if start > end:
        return rms_values

    return rms_values[start:end + 1]


def rms_to_centroid_ratio(rms_values: list[float]) -> float:
    if not rms_values:
        return 0.0

    total = sum(rms_values)
    if total == 0:
        return 0.0

    weighted = sum(index * value for index, value in enumerate(rms_values))
    center = weighted / total
    return center / max(1, len(rms_values) - 1)


def mean_slice(values: list[float], start: int, end: int) -> float:
    part = values[start:end]
    if not part:
        return 0.0
    return statistics.fmean(part)


def extract_features(path: Path) -> SampleFeatures:
    rms_values, sample_rate = load_rms_windows(path)
    active = trim_silence(rms_values)
    peak = max(active) if active else 0.0
    mean_rms = statistics.fmean(active) if active else 0.0
    peak_index = active.index(peak) if active else 0
    window_s = WINDOW_MS / 1000

    half = len(active) // 2
    quarter = max(1, len(active) // 4)
    sustain_mean = mean_slice(active, half, len(active))
    tail_mean = mean_slice(active, len(active) - quarter, len(active))

    label, note = parse_label_and_note(path)

    return SampleFeatures(
        filename=path.name,
        label=label,
        note=note,
        duration_s=len(active) * window_s,
        attack_time_s=peak_index * window_s,
        sustain_ratio=(sustain_mean / peak) if peak else 0.0,
        tail_ratio=(tail_mean / peak) if peak else 0.0,
        peak_mean_ratio=(peak / mean_rms) if mean_rms else 0.0,
        centroid_ratio=rms_to_centroid_ratio(active),
    )


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def standardize(train_vectors: list[list[float]], test_vector: list[float]) -> tuple[list[list[float]], list[float]]:
    dims = len(train_vectors[0])
    means = []
    stdevs = []

    for i in range(dims):
        column = [vector[i] for vector in train_vectors]
        mean = statistics.fmean(column)
        stdev = statistics.pstdev(column) or 1.0
        means.append(mean)
        stdevs.append(stdev)

    scaled_train = [
        [(value - means[i]) / stdevs[i] for i, value in enumerate(vector)]
        for vector in train_vectors
    ]
    scaled_test = [(value - means[i]) / stdevs[i] for i, value in enumerate(test_vector)]
    return scaled_train, scaled_test


def nearest_centroid_predict(train_items: list[SampleFeatures], test_item: SampleFeatures) -> str:
    distances = centroid_distances(train_items, test_item)
    return min(distances, key=distances.get)


def centroid_distances(train_items: list[SampleFeatures], test_item: SampleFeatures) -> dict[str, float]:
    train_vectors = [item.vector() for item in train_items]
    scaled_train, scaled_test = standardize(train_vectors, test_item.vector())

    grouped: dict[str, list[list[float]]] = {}
    for item, vector in zip(train_items, scaled_train):
        grouped.setdefault(item.label, []).append(vector)

    centroids: dict[str, list[float]] = {}
    for label, vectors in grouped.items():
        centroids[label] = [
            statistics.fmean(values)
            for values in zip(*vectors)
        ]

    return {
        label: euclidean_distance(scaled_test, centroids[label])
        for label in centroids
    }


def leave_one_out_accuracy(items: list[SampleFeatures]) -> tuple[float, list[tuple[str, str, str]]]:
    results: list[tuple[str, str, str]] = []

    for index, test_item in enumerate(items):
        train_items = items[:index] + items[index + 1:]
        predicted = nearest_centroid_predict(train_items, test_item)
        results.append((test_item.filename, test_item.label, predicted))

    correct = sum(1 for _, actual, predicted in results if actual == predicted)
    return correct / len(results), results


def print_feature_summary(items: list[SampleFeatures]) -> None:
    print("Feature summary by class")
    print(
        "label   attack_s  sustain_ratio  tail_ratio  peak_mean_ratio  centroid_ratio"
    )

    for label in ("piano", "organ"):
        label_items = [item for item in items if item.label == label]
        print(
            f"{label:5}  "
            f"{statistics.fmean(item.attack_time_s for item in label_items):8.3f}  "
            f"{statistics.fmean(item.sustain_ratio for item in label_items):13.3f}  "
            f"{statistics.fmean(item.tail_ratio for item in label_items):10.3f}  "
            f"{statistics.fmean(item.peak_mean_ratio for item in label_items):15.3f}  "
            f"{statistics.fmean(item.centroid_ratio for item in label_items):14.3f}"
        )


def print_samples(items: list[SampleFeatures]) -> None:
    print()
    print("Per-file features")
    print(
        "file            label   note  attack_s  sustain_ratio  tail_ratio  peak_mean_ratio  centroid_ratio"
    )
    for item in items:
        print(
            f"{item.filename:14}  {item.label:5}  {item.note:3}  "
            f"{item.attack_time_s:8.3f}  {item.sustain_ratio:13.3f}  "
            f"{item.tail_ratio:10.3f}  {item.peak_mean_ratio:15.3f}  "
            f"{item.centroid_ratio:14.3f}"
        )


def print_evaluation(items: list[SampleFeatures]) -> None:
    accuracy, results = leave_one_out_accuracy(items)
    print()
    print(f"Leave-one-out accuracy: {accuracy:.1%}")

    mistakes = [row for row in results if row[1] != row[2]]
    if mistakes:
        print("Misclassified files")
        for filename, actual, predicted in mistakes:
            print(f"{filename}: actual={actual}, predicted={predicted}")
    else:
        print("All files were classified correctly.")


def print_single_prediction(train_items: list[SampleFeatures], sample_path: Path) -> None:
    test_item = extract_features(sample_path)
    distances = centroid_distances(train_items, test_item)
    predicted = min(distances, key=distances.get)

    print(f"Input file: {sample_path}")
    print(f"Predicted label: {predicted}")
    print(
        "Features: "
        f"attack_s={test_item.attack_time_s:.3f}, "
        f"sustain_ratio={test_item.sustain_ratio:.3f}, "
        f"tail_ratio={test_item.tail_ratio:.3f}, "
        f"peak_mean_ratio={test_item.peak_mean_ratio:.3f}, "
        f"centroid_ratio={test_item.centroid_ratio:.3f}"
    )
    print(
        "Distances: "
        + ", ".join(f"{label}={distance:.3f}" for label, distance in sorted(distances.items()))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from wav files and classify piano vs organ."
    )
    parser.add_argument(
        "sample",
        nargs="?",
        help="Optional path to a wav file to classify.",
    )
    return parser.parse_args()


def load_training_items() -> list[SampleFeatures]:
    wav_files = sorted(DATA_DIR.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No wav files found in {DATA_DIR}")

    items = [extract_features(path) for path in wav_files]
    filtered = [item for item in items if item.label in {"piano", "organ"}]
    if not filtered:
        raise FileNotFoundError(f"No piano/organ training files found in {DATA_DIR}")
    return filtered


def main() -> None:
    args = parse_args()
    items = load_training_items()
    if args.sample:
        sample_path = Path(args.sample)
        if not sample_path.is_absolute():
            sample_path = (Path.cwd() / sample_path).resolve()
        if sample_path.suffix.lower() != ".wav":
            raise ValueError("Only .wav files are supported for classification.")
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")
        print_single_prediction(items, sample_path)
        return

    print_feature_summary(items)
    print_samples(items)
    print_evaluation(items)


if __name__ == "__main__":
    main()

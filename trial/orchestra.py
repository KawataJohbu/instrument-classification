from __future__ import annotations

import argparse
import cmath
import math
import statistics
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "wavlist" / "URMP_Sep"
WINDOW_MS = 50
DEFAULT_MAX_SECONDS = 15.0
DEFAULT_SEGMENT_SECONDS = 2.0
MIN_SEGMENT_SECONDS = 0.75
MIN_SEGMENT_RMS_RATIO = 0.05
FFT_SIZE = 1024
MAX_SPECTRAL_FRAMES = 24
MFCC_COUNT = 13
CLASS_NAMES = ("violin", "cello", "flute", "oboe", "horn")
DEFAULT_PRESENCE_THRESHOLD = 20.0
SCORE_TEMPERATURE = 1.0
DEFAULT_MODEL = "sklearn"


@dataclass
class SampleFeatures:
    filename: str
    label: str
    source: str
    segment_start_s: float
    duration_s: float
    attack_time_s: float
    sustain_ratio: float
    tail_ratio: float
    peak_mean_ratio: float
    centroid_ratio: float
    spectral_centroid_ratio: float
    mfcc_values: tuple[float, ...]

    def vector(self) -> list[float]:
        return [
            self.attack_time_s,
            self.sustain_ratio,
            self.tail_ratio,
            self.peak_mean_ratio,
            self.centroid_ratio,
            self.spectral_centroid_ratio,
        ] + list(self.mfcc_values)


def infer_label_and_source(path: Path) -> tuple[str, str]:
    parent_name = path.parent.name.lower()
    label = parent_name if parent_name in CLASS_NAMES else "unknown"
    return label, path.stem


def load_mono_samples(
    path: Path,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav_file:
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
        if max_seconds is not None and max_seconds > 0:
            frame_count = min(frame_count, int(sample_rate * max_seconds))
        raw = wav_file.readframes(frame_count)

    mono_samples = decode_mono_samples(raw, sample_width, channels)
    return sample_rate, mono_samples


def rms_windows_from_samples(
    mono_samples: list[float],
    sample_rate: int,
    window_ms: int = WINDOW_MS,
) -> list[float]:
    samples_per_window = max(1, int(sample_rate * window_ms / 1000))
    rms_values: list[float] = []

    for start in range(0, len(mono_samples) - samples_per_window + 1, samples_per_window):
        chunk = mono_samples[start:start + samples_per_window]
        rms_values.append(calculate_rms(chunk))

    return rms_values


def load_rms_windows(
    path: Path,
    window_ms: int = WINDOW_MS,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
) -> tuple[list[float], int, list[float]]:
    sample_rate, mono_samples = load_mono_samples(path, max_seconds=max_seconds)
    rms_values = rms_windows_from_samples(mono_samples, sample_rate, window_ms=window_ms)
    return rms_values, sample_rate, mono_samples


def decode_mono_samples(raw: bytes, sample_width: int, channels: int) -> list[float]:
    if sample_width == 1:
        values = [sample - 128 for sample in raw]
    elif sample_width == 3:
        values = []
        for index in range(0, len(raw), 3):
            chunk = raw[index:index + 3]
            if len(chunk) < 3:
                continue
            sign_byte = b"\xff" if chunk[2] & 0x80 else b"\x00"
            values.append(int.from_bytes(chunk + sign_byte, byteorder="little", signed=True))
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


def fft(values: list[complex]) -> list[complex]:
    size = len(values)
    if size <= 1:
        return values

    even = fft(values[0::2])
    odd = fft(values[1::2])
    result = [0j] * size

    for index in range(size // 2):
        twiddle = cmath.exp(-2j * math.pi * index / size) * odd[index]
        result[index] = even[index] + twiddle
        result[index + size // 2] = even[index] - twiddle

    return result


def select_frame_starts(sample_count: int, frame_size: int, max_frames: int) -> list[int]:
    if sample_count < frame_size:
        return []

    possible_count = sample_count - frame_size + 1
    if max_frames <= 1:
        return [0]

    if possible_count <= max_frames:
        return list(range(possible_count))

    step = (possible_count - 1) / (max_frames - 1)
    return [round(step * index) for index in range(max_frames)]


def spectral_centroid_ratio(samples: list[float], sample_rate: int) -> float:
    starts = select_frame_starts(len(samples), FFT_SIZE, MAX_SPECTRAL_FRAMES)
    if not starts:
        return 0.0

    window = [
        0.5 - 0.5 * math.cos(2 * math.pi * index / (FFT_SIZE - 1))
        for index in range(FFT_SIZE)
    ]
    centroids: list[float] = []

    for start in starts:
        frame = samples[start:start + FFT_SIZE]
        mean = statistics.fmean(frame)
        spectrum = fft([
            complex((sample - mean) * window[index], 0.0)
            for index, sample in enumerate(frame)
        ])

        weighted_sum = 0.0
        magnitude_sum = 0.0
        for bin_index in range(1, FFT_SIZE // 2 + 1):
            magnitude = abs(spectrum[bin_index])
            frequency = bin_index * sample_rate / FFT_SIZE
            weighted_sum += frequency * magnitude
            magnitude_sum += magnitude

        if magnitude_sum > 0:
            centroid_hz = weighted_sum / magnitude_sum
            centroids.append(centroid_hz / (sample_rate / 2))

    if not centroids:
        return 0.0
    return statistics.fmean(centroids)


def mfcc_features(samples: list[float], sample_rate: int) -> tuple[float, ...]:
    if not samples:
        return (0.0,) * (MFCC_COUNT * 2)

    try:
        import librosa
        import numpy as np
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "librosa is required for MFCC features. "
            "Install it with: .\\.venv\\Scripts\\python.exe -m pip install librosa"
        ) from error

    max_abs = max(abs(sample) for sample in samples) or 1.0
    waveform = np.asarray(samples, dtype=np.float32) / max_abs
    n_fft = min(2048, max(256, len(waveform)))

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=MFCC_COUNT,
        n_fft=n_fft,
        hop_length=512,
    )
    means = mfcc.mean(axis=1)
    stdevs = mfcc.std(axis=1)
    return tuple(float(value) for value in np.concatenate([means, stdevs]))


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


def extract_features_from_samples(
    path: Path,
    mono_samples: list[float],
    sample_rate: int,
    segment_start_s: float = 0.0,
) -> SampleFeatures:
    rms_values = rms_windows_from_samples(mono_samples, sample_rate)
    active = trim_silence(rms_values)
    peak = max(active) if active else 0.0
    mean_rms = statistics.fmean(active) if active else 0.0
    peak_index = active.index(peak) if active else 0
    window_s = WINDOW_MS / 1000

    half = len(active) // 2
    quarter = max(1, len(active) // 4)
    sustain_mean = mean_slice(active, half, len(active))
    tail_mean = mean_slice(active, len(active) - quarter, len(active))

    label, source = infer_label_and_source(path)

    return SampleFeatures(
        filename=path.name,
        label=label,
        source=source,
        segment_start_s=segment_start_s,
        duration_s=len(active) * window_s,
        attack_time_s=peak_index * window_s,
        sustain_ratio=(sustain_mean / peak) if peak else 0.0,
        tail_ratio=(tail_mean / peak) if peak else 0.0,
        peak_mean_ratio=(peak / mean_rms) if mean_rms else 0.0,
        centroid_ratio=rms_to_centroid_ratio(active),
        spectral_centroid_ratio=spectral_centroid_ratio(mono_samples, sample_rate),
        mfcc_values=mfcc_features(mono_samples, sample_rate),
    )


def extract_features(path: Path, max_seconds: float | None = DEFAULT_MAX_SECONDS) -> SampleFeatures:
    sample_rate, mono_samples = load_mono_samples(path, max_seconds=max_seconds)
    return extract_features_from_samples(path, mono_samples, sample_rate)


def extract_segment_features(
    path: Path,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
) -> list[SampleFeatures]:
    sample_rate, mono_samples = load_mono_samples(path, max_seconds=max_seconds)
    segment_size = max(1, int(sample_rate * segment_seconds))
    min_segment_size = max(1, int(sample_rate * MIN_SEGMENT_SECONDS))

    segments: list[tuple[int, list[float], float]] = []
    for start in range(0, len(mono_samples), segment_size):
        segment = mono_samples[start:start + segment_size]
        if len(segment) < min_segment_size:
            continue
        segments.append((start, segment, calculate_rms(segment)))

    if not segments:
        return [extract_features_from_samples(path, mono_samples, sample_rate)]

    max_rms = max(rms for _start, _segment, rms in segments)
    threshold = max_rms * MIN_SEGMENT_RMS_RATIO
    features: list[SampleFeatures] = []

    for start, segment, rms in segments:
        if max_rms > 0 and rms < threshold:
            continue
        segment_start_s = start / sample_rate
        features.append(extract_features_from_samples(path, segment, sample_rate, segment_start_s))

    if not features:
        return [extract_features_from_samples(path, mono_samples, sample_rate)]
    return features


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


def centroid_distances(train_items: list[SampleFeatures], test_item: SampleFeatures) -> dict[str, float]:
    train_vectors = [item.vector() for item in train_items]
    scaled_train, scaled_test = standardize(train_vectors, test_item.vector())

    grouped: dict[str, list[list[float]]] = {}
    for item, vector in zip(train_items, scaled_train):
        grouped.setdefault(item.label, []).append(vector)

    centroids: dict[str, list[float]] = {}
    for label, vectors in grouped.items():
        centroids[label] = [statistics.fmean(values) for values in zip(*vectors)]

    return {
        label: euclidean_distance(scaled_test, centroids[label])
        for label in centroids
    }


def average_segment_distances(
    train_items: list[SampleFeatures],
    test_items: list[SampleFeatures],
) -> dict[str, float]:
    segment_distances = [centroid_distances(train_items, item) for item in test_items]
    labels = sorted({label for distances in segment_distances for label in distances})

    return {
        label: statistics.fmean(distances[label] for distances in segment_distances)
        for label in labels
    }


def distances_to_similarity_scores(distances: dict[str, float]) -> dict[str, float]:
    similarities = {
        label: math.exp(-distance / SCORE_TEMPERATURE)
        for label, distance in distances.items()
    }
    total_similarity = sum(similarities.values()) or 1.0

    return {
        label: (similarity / total_similarity) * 100.0
        for label, similarity in similarities.items()
    }


def average_segment_scores(
    train_items: list[SampleFeatures],
    test_items: list[SampleFeatures],
) -> tuple[dict[str, float], dict[str, int]]:
    segment_scores: list[dict[str, float]] = []
    votes = {label: 0 for label in CLASS_NAMES}

    for item in test_items:
        distances = centroid_distances(train_items, item)
        predicted = min(distances, key=distances.get)
        if predicted in votes:
            votes[predicted] += 1
        segment_scores.append(distances_to_similarity_scores(distances))

    labels = sorted({label for scores in segment_scores for label in scores})
    scores = {
        label: statistics.fmean(scores[label] for scores in segment_scores)
        for label in labels
    }

    return scores, votes


def import_sklearn_tools():
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "scikit-learn is required for --model sklearn. "
            "Install it with: .\\.venv\\Scripts\\python.exe -m pip install scikit-learn"
        ) from error

    return make_pipeline, StandardScaler, SVC


def train_sklearn_model(train_items: list[SampleFeatures]):
    make_pipeline, StandardScaler, SVC = import_sklearn_tools()
    vectors = [item.vector() for item in train_items]
    labels = [item.label for item in train_items]

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=3.0,
            gamma="scale",
            class_weight="balanced",
            random_state=0,
        ),
    )
    model.fit(vectors, labels)
    return model


def sklearn_file_scores(model, test_items: list[SampleFeatures]) -> tuple[dict[str, float], dict[str, int]]:
    vectors = [item.vector() for item in test_items]
    decision_values = model.decision_function(vectors)
    predictions = model.predict(vectors)
    classes = list(model.classes_)

    if len(classes) == 2:
        decision_values = [[-float(value), float(value)] for value in decision_values]

    probabilities: list[list[float]] = []
    for row in decision_values:
        max_value = max(float(value) for value in row)
        exp_values = [math.exp(float(value) - max_value) for value in row]
        total = sum(exp_values) or 1.0
        probabilities.append([value / total for value in exp_values])

    scores = {
        label: statistics.fmean(float(row[index]) for row in probabilities) * 100.0
        for index, label in enumerate(classes)
    }
    votes = {label: 0 for label in CLASS_NAMES}
    for prediction in predictions:
        if prediction in votes:
            votes[str(prediction)] += 1

    return scores, votes


def sklearn_predict_file(model, test_items: list[SampleFeatures]) -> str:
    scores, _votes = sklearn_file_scores(model, test_items)
    return max(scores, key=scores.get)


def nearest_centroid_predict(train_items: list[SampleFeatures], test_item: SampleFeatures) -> str:
    distances = centroid_distances(train_items, test_item)
    return min(distances, key=distances.get)


def group_items_by_file(items: list[SampleFeatures]) -> dict[str, list[SampleFeatures]]:
    grouped: dict[str, list[SampleFeatures]] = {}
    for item in items:
        grouped.setdefault(item.source, []).append(item)
    return grouped


def leave_one_file_out_accuracy(
    items: list[SampleFeatures],
    model_name: str = DEFAULT_MODEL,
) -> tuple[float, list[tuple[str, str, str]]]:
    results: list[tuple[str, str, str]] = []
    grouped = group_items_by_file(items)

    for source, test_items in grouped.items():
        train_items = [item for item in items if item.source != source]
        if model_name == "sklearn":
            model = train_sklearn_model(train_items)
            predicted = sklearn_predict_file(model, test_items)
        else:
            distances = average_segment_distances(train_items, test_items)
            predicted = min(distances, key=distances.get)
        actual = test_items[0].label
        results.append((test_items[0].filename, actual, predicted))

    correct = sum(1 for _, actual, predicted in results if actual == predicted)
    return correct / len(results), results


def print_feature_summary(items: list[SampleFeatures]) -> None:
    print("Feature summary by class")
    print("label    files  segments  attack_s  sustain_ratio  tail_ratio  peak_mean_ratio  centroid_ratio  spectral_centroid  mfcc1_mean")

    for label in CLASS_NAMES:
        label_items = [item for item in items if item.label == label]
        if not label_items:
            continue
        file_count = len({item.source for item in label_items})

        print(
            f"{label:7}  "
            f"{file_count:5d}  "
            f"{len(label_items):8d}  "
            f"{statistics.fmean(item.attack_time_s for item in label_items):8.3f}  "
            f"{statistics.fmean(item.sustain_ratio for item in label_items):13.3f}  "
            f"{statistics.fmean(item.tail_ratio for item in label_items):10.3f}  "
            f"{statistics.fmean(item.peak_mean_ratio for item in label_items):15.3f}  "
            f"{statistics.fmean(item.centroid_ratio for item in label_items):14.3f}  "
            f"{statistics.fmean(item.spectral_centroid_ratio for item in label_items):17.3f}  "
            f"{statistics.fmean(item.mfcc_values[0] for item in label_items):10.3f}"
        )


def print_samples(items: list[SampleFeatures]) -> None:
    print()
    print("Per-file features")
    print("file                                              label    start_s  attack_s  sustain_ratio  tail_ratio  peak_mean_ratio  centroid_ratio  spectral_centroid  mfcc1_mean")
    for item in items:
        print(
            f"{item.filename:48}  {item.label:7}  "
            f"{item.segment_start_s:7.2f}  "
            f"{item.attack_time_s:8.3f}  {item.sustain_ratio:13.3f}  "
            f"{item.tail_ratio:10.3f}  {item.peak_mean_ratio:15.3f}  "
            f"{item.centroid_ratio:14.3f}  {item.spectral_centroid_ratio:17.3f}  "
            f"{item.mfcc_values[0]:10.3f}"
        )


def print_evaluation(items: list[SampleFeatures], model_name: str = DEFAULT_MODEL) -> None:
    accuracy, results = leave_one_file_out_accuracy(items, model_name=model_name)
    print()
    print(f"Model: {model_name}")
    print(f"Leave-one-file-out accuracy: {accuracy:.1%}")

    mistakes = [row for row in results if row[1] != row[2]]
    if mistakes:
        print("Misclassified files")
        for filename, actual, predicted in mistakes:
            print(f"{filename}: actual={actual}, predicted={predicted}")
    else:
        print("All files were classified correctly.")


def print_single_prediction(
    train_items: list[SampleFeatures],
    sample_path: Path,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    model_name: str = DEFAULT_MODEL,
) -> None:
    test_items = extract_segment_features(
        sample_path,
        segment_seconds=segment_seconds,
        max_seconds=max_seconds,
    )
    if model_name == "sklearn":
        model = train_sklearn_model(train_items)
        scores, _votes = sklearn_file_scores(model, test_items)
        predicted = max(scores, key=scores.get)
    else:
        distances = average_segment_distances(train_items, test_items)
        predicted = min(distances, key=distances.get)
        scores = distances_to_similarity_scores(distances)
    first_item = test_items[0]

    print(f"Input file: {sample_path}")
    print(f"Model: {model_name}")
    print(f"Predicted label: {predicted}")
    print(f"Segments analyzed: {len(test_items)}")
    print(
        "First segment features: "
        f"attack_s={first_item.attack_time_s:.3f}, "
        f"sustain_ratio={first_item.sustain_ratio:.3f}, "
        f"tail_ratio={first_item.tail_ratio:.3f}, "
        f"peak_mean_ratio={first_item.peak_mean_ratio:.3f}, "
        f"centroid_ratio={first_item.centroid_ratio:.3f}, "
        f"spectral_centroid_ratio={first_item.spectral_centroid_ratio:.3f}, "
        f"mfcc1_mean={first_item.mfcc_values[0]:.3f}"
    )
    print("Scores: " + ", ".join(f"{label}={score:.1f}" for label, score in sorted(scores.items())))


def print_multi_label_prediction(
    train_items: list[SampleFeatures],
    sample_path: Path,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    threshold: float = DEFAULT_PRESENCE_THRESHOLD,
    model_name: str = DEFAULT_MODEL,
) -> None:
    test_items = extract_segment_features(
        sample_path,
        segment_seconds=segment_seconds,
        max_seconds=max_seconds,
    )
    if model_name == "sklearn":
        model = train_sklearn_model(train_items)
        scores, votes = sklearn_file_scores(model, test_items)
    else:
        scores, votes = average_segment_scores(train_items, test_items)
    ranked_scores = sorted(scores.items(), key=lambda row: row[1], reverse=True)
    detected = [(label, score) for label, score in ranked_scores if score >= threshold]

    print(f"Input file: {sample_path}")
    print(f"Model: {model_name}")
    print(f"Segments analyzed: {len(test_items)}")
    print("Relative instrument scores")
    for label, score in ranked_scores:
        print(f"{label:7}: {score:6.1f} / 100  segment_votes={votes.get(label, 0)}")

    print()
    if detected:
        print(
            f"Detected instruments (threshold >= {threshold:.1f}): "
            + ", ".join(label for label, _score in detected)
        )
    else:
        print(
            f"No instrument reached threshold >= {threshold:.1f}. "
            f"Top candidate: {ranked_scores[0][0]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from URMP Sep wav files and classify instruments."
    )
    parser.add_argument(
        "sample",
        nargs="?",
        help="Optional path to a wav file to classify.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Training data root. Expect subfolders such as violin/cello/flute/oboe/horn.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_MAX_SECONDS,
        help="Seconds analyzed from the start of each wav. Use 0 to analyze full files.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=DEFAULT_SEGMENT_SECONDS,
        help="Seconds per analysis segment.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-file features.",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Print a score for each instrument instead of only one predicted label.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PRESENCE_THRESHOLD,
        help="Score threshold used by --multi-label. Scores are relative values from 0 to 100.",
    )
    parser.add_argument(
        "--model",
        choices=("sklearn", "centroid"),
        default=DEFAULT_MODEL,
        help="Classifier to use. sklearn uses SVC; centroid uses the earlier nearest-centroid method.",
    )
    return parser.parse_args()


def normalize_max_seconds(value: float) -> float | None:
    return None if value <= 0 else value


def load_training_items(
    data_dir: Path,
    max_seconds: float | None,
    segment_seconds: float,
) -> list[SampleFeatures]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data folder was not found: {data_dir}")

    items: list[SampleFeatures] = []
    for label in CLASS_NAMES:
        class_dir = data_dir / label
        if not class_dir.exists():
            continue
        for wav_path in sorted(class_dir.glob("*.wav")):
            items.extend(
                extract_segment_features(
                    wav_path,
                    segment_seconds=segment_seconds,
                    max_seconds=max_seconds,
                )
            )

    if not items:
        raise FileNotFoundError(f"No training wav files found in {data_dir}")
    return items


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()

    max_seconds = normalize_max_seconds(args.max_seconds)
    items = load_training_items(
        data_dir,
        max_seconds=max_seconds,
        segment_seconds=args.segment_seconds,
    )

    if args.sample:
        sample_path = Path(args.sample)
        if not sample_path.is_absolute():
            sample_path = (Path.cwd() / sample_path).resolve()
        if sample_path.suffix.lower() != ".wav":
            raise ValueError("Only .wav files are supported for classification.")
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")
        if args.multi_label:
            print_multi_label_prediction(
                items,
                sample_path,
                max_seconds=max_seconds,
                segment_seconds=args.segment_seconds,
                threshold=args.threshold,
                model_name=args.model,
            )
            return
        print_single_prediction(
            items,
            sample_path,
            max_seconds=max_seconds,
            segment_seconds=args.segment_seconds,
            model_name=args.model,
        )
        return

    print_feature_summary(items)
    if args.details:
        print_samples(items)
    print_evaluation(items, model_name=args.model)


if __name__ == "__main__":
    main()

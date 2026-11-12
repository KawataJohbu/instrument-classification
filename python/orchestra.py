from __future__ import annotations

import argparse
import ast
import cmath
import csv
import hashlib
import json
import math
import statistics
import struct
import wave
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "wavlist" / "URMP_Sep"
WINDOW_MS = 50
DEFAULT_MAX_SECONDS = 120.0
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
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "orchestra"
MODEL_CACHE = Path(__file__).resolve().parent / ".cache" / "orchestra_live.joblib"


def learning_source_hash() -> str:
    """表示や評価処理の変更では学習済みモデルを無効にしない。"""
    names = {
        "SampleFeatures", "infer_label_and_source", "load_mono_samples",
        "rms_windows_from_samples", "decode_mono_samples", "calculate_rms",
        "fft", "select_frame_starts", "spectral_centroid_ratio", "mfcc_features",
        "trim_silence", "rms_to_centroid_ratio", "mean_slice",
        "extract_features_from_samples", "extract_segment_features",
        "import_sklearn_tools", "train_sklearn_model", "load_training_items",
    }
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8-sig"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(ast.dump(node, include_attributes=False))
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.isupper()
            and target.id not in {"PROJECT_ROOT", "DATA_DIR", "CACHE_DIR", "MODEL_CACHE"}
            for target in node.targets
        ):
            nodes.append(ast.dump(node, include_attributes=False))
    return hashlib.sha256("\n".join(nodes).encode()).hexdigest()


def load_or_train_shared_model(data_dir, max_seconds, segment_seconds, retrain=False,
                               cache_path=None):
    import joblib
    import sklearn
    import numpy

    path = MODEL_CACHE if cache_path is None else Path(cache_path)
    data_dir = Path(data_dir).resolve()
    files = []
    for label in CLASS_NAMES:
        for wav in sorted((data_dir / label).glob("*.wav")):
            info = wav.stat()
            files.append((str(wav.relative_to(data_dir)), info.st_size, info.st_mtime_ns))
    if not files:
        raise FileNotFoundError(f"No training wav files found in {data_dir}")
    signature = {
        "data_dir": str(data_dir), "files": files, "max_seconds": max_seconds,
        "segment_seconds": segment_seconds, "source": learning_source_hash(),
        "sklearn": sklearn.__version__, "numpy": numpy.__version__, "cache_version": 2,
        "librosa": version("librosa"),
    }
    if path.exists() and not retrain:
        try:
            cached = joblib.load(path)
            previous = cached["signature"]
            # この変更前に保存済みの、学習ロジックが同一の既知バージョンを引き継ぐ。
            legacy_hashes = {
                "53e104e1bbb3bb8e4534b3c1ac9531f7e6d193f9d59bb92b2785e636f25c5679",
                "92b439c42e39aad7a10841f79cbbc30675e7e73d0860ce57a7f30f54ca531411",
            }
            legacy = (previous.get("cache_version") == 1
                      and previous.get("source") in legacy_hashes
                      and all(previous.get(key) == value for key, value in signature.items()
                              if key not in {"source", "cache_version", "librosa"}))
            if previous == signature or legacy:
                model = cached["model"]
                if legacy:
                    save_shared_model(path, signature, model)
                print("Loaded saved model. Skipping training.")
                return model
            print("Training data, learning logic or settings changed. Rebuilding model...")
        except Exception as error:
            print(f"Could not load saved model ({type(error).__name__}). Rebuilding...")
    key = training_cache_key(data_dir, max_seconds, segment_seconds)
    items = load_cached_training_items(data_dir, max_seconds, segment_seconds, key, rebuild=retrain)
    print(f"Training segments: {len(items)}")
    print("Training sklearn model...")
    model = train_sklearn_model(items)
    save_shared_model(path, signature, model)
    return model


def save_shared_model(path, signature, model):
    import joblib
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
        joblib.dump({"signature": signature, "model": model}, temporary)
        temporary.replace(path)
        print("Saved model for next startup.")
    except OSError as error:
        print(f"Could not save model ({type(error).__name__}); continuing.")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def training_cache_key(data_dir: Path, max_seconds: float | None, segment_seconds: float) -> str:
    data_dir = data_dir.resolve()
    files = []
    for label in CLASS_NAMES:
        for path in sorted((data_dir / label).glob("*.wav")):
            info = path.stat()
            files.append((str(path.relative_to(data_dir)), info.st_size, info.st_mtime_ns))
    if not files:
        raise FileNotFoundError(f"No training wav files found in {data_dir}")
    signature = {
        "directory": str(data_dir), "files": files,
        "max_seconds": max_seconds, "segment_seconds": segment_seconds,
        "source": learning_source_hash(),
        "versions": {name: version(name) for name in ("numpy", "librosa", "scikit-learn")},
    }
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def read_cache(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        print(f"Could not read cache ({type(error).__name__}); rebuilding.")
        return None


def save_cache(path: Path, value) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output)
        temporary.replace(path)
    except OSError as error:
        print(f"Could not save cache ({type(error).__name__}); continuing.")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_cached_training_items(data_dir, max_seconds, segment_seconds, key, rebuild=False):
    path = CACHE_DIR / f"features-{key}.json"
    cached = None if rebuild else read_cache(path)
    if cached is not None:
        try:
            items = [SampleFeatures(**row) for row in cached]
            if items:
                print("Loaded saved features. Skipping audio analysis.")
                return items
        except (TypeError, ValueError):
            print("Invalid feature cache; rebuilding.")
    print("Analyzing training audio...")
    items = load_training_items(data_dir, max_seconds, segment_seconds)
    save_cache(path, [asdict(item) for item in items])
    return items


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


def print_evaluation(items: list[SampleFeatures], model_name: str = DEFAULT_MODEL,
                     cache_key: str | None = None, rebuild: bool = False) -> None:
    path = CACHE_DIR / f"evaluation-{model_name}-{cache_key}.json" if cache_key else None
    cached = read_cache(path) if path is not None and not rebuild else None
    if (isinstance(cached, dict) and isinstance(cached.get("accuracy"), (int, float))
            and isinstance(cached.get("results"), list)
            and all(isinstance(row, list) and len(row) == 3 for row in cached["results"])):
        accuracy, results = cached["accuracy"], cached["results"]
        print("Loaded saved evaluation. Skipping repeated training.")
    else:
        print("Evaluating with leave-one-file-out training...")
        accuracy, results = leave_one_file_out_accuracy(items, model_name=model_name)
        if path is not None:
            save_cache(path, {"accuracy": accuracy, "results": results})
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
    model=None,
) -> None:
    test_items = extract_segment_features(
        sample_path,
        segment_seconds=segment_seconds,
        max_seconds=max_seconds,
    )
    if model_name == "sklearn":
        if model is None:
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
    model=None,
) -> None:
    test_items = extract_segment_features(
        sample_path,
        segment_seconds=segment_seconds,
        max_seconds=max_seconds,
    )
    if model_name == "sklearn":
        if model is None:
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


def evaluate_mix(model, test_dir: Path, max_seconds, segment_seconds, threshold,
                 rebuild=False):
    """Mixは推論専用。正解は収集時に作成したmetadata.csvから読む。"""
    metadata = test_dir / "metadata.csv"
    with metadata.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"filename", "instruments"}.issubset(reader.fieldnames or []):
            raise ValueError("metadata.csv requires filename and instruments columns")
        rows = list(reader)
    if not rows:
        raise ValueError("No test files in metadata.csv")
    tests = []
    seen = set()
    for row in rows:
        path = (test_dir / row["filename"]).resolve()
        if path.parent != test_dir.resolve() or path.suffix.lower() != ".wav":
            raise ValueError(f"Invalid test filename: {row['filename']}")
        if path.name in seen:
            raise ValueError(f"Duplicate test file: {path.name}")
        seen.add(path.name)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = set(filter(None, (label.strip() for label in row["instruments"].split(";"))))
        if actual - set(CLASS_NAMES):
            raise ValueError(f"Unknown labels in metadata: {actual}")
        tests.append((path, actual))
    counts = {label: [0, 0, 0] for label in CLASS_NAMES}  # TP, FP, FN
    exact = 0
    print("\nTest dataset: URMP_Mix (inference only)")
    print(f"Target labels: {', '.join(CLASS_NAMES)}; threshold: {threshold:g}/100")
    print("Other instruments and the number of players are outside this evaluation.")
    print("NOTE: Mix/Sep may contain the same performances; this is not an independent-song test.")
    for index, (path, actual) in enumerate(tests, 1):
        info = path.stat()
        signature = (str(path), info.st_size, info.st_mtime_ns, max_seconds,
                     segment_seconds, learning_source_hash(), version("librosa"), version("numpy"))
        key = hashlib.sha256(repr(signature).encode()).hexdigest()
        cache = CACHE_DIR / f"mix-features-{key}.json"
        stored = None if rebuild else read_cache(cache)
        features = None
        if stored:
            try:
                features = [SampleFeatures(**row) for row in stored]
            except (TypeError, ValueError):
                pass
        if not features:
            print(f"[{index}/{len(tests)}] Analyzing {path.name}...", flush=True)
            features = extract_segment_features(path, segment_seconds, max_seconds)
            save_cache(cache, [asdict(item) for item in features])
        scores, _ = sklearn_file_scores(model, features)
        predicted = {label for label, score in scores.items() if score >= threshold}
        matched = actual == predicted
        exact += matched
        for label in CLASS_NAMES:
            counts[label][0] += label in actual and label in predicted
            counts[label][1] += label not in actual and label in predicted
            counts[label][2] += label in actual and label not in predicted
        print(f"[{index}/{len(tests)}] {'OK' if matched else 'MISS'} {path.name}")
        print(f"  actual={','.join(sorted(actual)) or '(none)'}; predicted={','.join(sorted(predicted)) or '(none)'}", flush=True)
    def metrics(tp, fp, fn):
        return (tp/(tp+fp) if tp+fp else 0., tp/(tp+fn) if tp+fn else 0.,
                2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    print(f"\nExact-match accuracy: {exact}/{len(tests)} = {exact/len(tests):.1%}")
    print("Instrument   Precision    Recall       F1")
    for label, values in counts.items():
        precision, recall, f1 = metrics(*values)
        print(f"{label:10}  {precision:9.1%}  {recall:8.1%}  {f1:8.1%}")
    totals = [sum(values[i] for values in counts.values()) for i in range(3)]
    precision, recall, f1 = metrics(*totals)
    print(f"Micro avg   {precision:9.1%}  {recall:8.1%}  {f1:8.1%}")
    return {"files": len(tests), "exact": exact, "precision": precision, "recall": recall, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from URMP Sep wav files and classify instruments."
    )
    parser.add_argument("--retrain", action="store_true",
                        help="Rebuild saved features and evaluation results.")
    parser.add_argument("--use-live-model", action="store_true",
                        help="Use the saved live model to measure TRAINING accuracy (not cross-validation).")
    parser.add_argument("--evaluation", choices=("mix", "leave-one-file-out"), default="mix",
                        help="Default: test the saved Sep model on URMP_Mix.")
    parser.add_argument("--test-dir", default=str(PROJECT_ROOT / "wavlist" / "URMP_Mix"),
                        help="Mix test folder containing wav files and metadata.csv.")
    parser.add_argument("--test-max-seconds", type=float,
                        help="Mix test duration only; 0 uses the whole file. Defaults to --max-seconds.")
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
    if not args.sample and not args.use_live_model and args.evaluation == "mix":
        if args.model != "sklearn":
            raise ValueError("Mix evaluation requires --model sklearn")
        if (not 0 <= args.threshold <= 100 or not math.isfinite(args.segment_seconds)
                or args.segment_seconds <= 0 or not math.isfinite(args.max_seconds)
                or (args.test_max_seconds is not None and not math.isfinite(args.test_max_seconds))):
            raise ValueError("threshold must be 0..100 and segment-seconds must be positive")
        test_dir = Path(args.test_dir)
        if not test_dir.is_absolute():
            test_dir = PROJECT_ROOT / test_dir
        if not (test_dir / "metadata.csv").is_file():
            raise FileNotFoundError(test_dir / "metadata.csv")
        model = load_or_train_shared_model(data_dir, max_seconds, args.segment_seconds,
                                          retrain=args.retrain)
        test_seconds = (max_seconds if args.test_max_seconds is None
                        else normalize_max_seconds(args.test_max_seconds))
        evaluate_mix(model, test_dir, test_seconds, args.segment_seconds, args.threshold,
                     rebuild=args.retrain)
        return
    if args.use_live_model and args.model != "sklearn":
        raise ValueError("--use-live-model requires --model sklearn")
    model = None
    if args.model == "sklearn" and (args.sample or args.use_live_model):
        model = load_or_train_shared_model(
            data_dir, max_seconds, args.segment_seconds, retrain=args.retrain,
        )
    cache_key = training_cache_key(data_dir, max_seconds, args.segment_seconds)
    items = [] if args.sample and model is not None else load_cached_training_items(
        data_dir, max_seconds, args.segment_seconds, cache_key, rebuild=args.retrain,
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
                model=model,
            )
            return
        print_single_prediction(
            items,
            sample_path,
            max_seconds=max_seconds,
            segment_seconds=args.segment_seconds,
            model_name=args.model,
            model=model,
        )
        return

    print_feature_summary(items)
    if args.details:
        print_samples(items)
    if args.use_live_model:
        results = []
        for test_items in group_items_by_file(items).values():
            results.append((test_items[0].filename, test_items[0].label,
                            sklearn_predict_file(model, test_items)))
        correct = sum(actual == predicted for _, actual, predicted in results)
        print("\nWARNING: These files were used to train this saved model.")
        print("This is training accuracy, NOT leave-one-file-out or microphone accuracy.")
        print(f"Training accuracy (saved live model): {correct / len(results):.1%}")
        for filename, actual, predicted in results:
            if actual != predicted:
                print(f"{filename}: actual={actual}, predicted={predicted}")
        return
    print_evaluation(items, model_name=args.model, cache_key=cache_key, rebuild=args.retrain)


if __name__ == "__main__":
    main()

from pathlib import Path
import os
import stat
import shutil

# URMP dataset root
URMP_ROOT = Path(r"C:\URMP")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Output directory for separated-part audio only
DATA_DIR = PROJECT_ROOT / "wavlist" / "URMP_Sep"

# Target URMP instrument codes
TARGET_INSTRUMENTS = {
    "vn": "violin",
    "vc": "cello",
    "fl": "flute",
    "ob": "oboe",
    "hn": "horn",
}

# Maximum number of files copied for each instrument
MAX_FILES_PER_INSTRUMENT = 30


def detect_instrument(file_name: str) -> str | None:
    lower_name = file_name.lower()

    for short_name, label in TARGET_INSTRUMENTS.items():
        if f"_{short_name}_" in lower_name or lower_name.startswith(f"{short_name}_"):
            return label

    return None


def is_target_ausep_file(wav_path: Path) -> bool:
    lower_name = wav_path.name.lower()

    # Skip hidden AppleDouble-style files such as ._xxx.wav
    if lower_name.startswith("._"):
        return False

    # Keep only separated-part audio
    return "ausep" in lower_name


def remove_path(target: Path) -> None:
    if target.is_dir():
        for child in target.iterdir():
            remove_path(child)
        os.chmod(target, stat.S_IWRITE)
        target.rmdir()
        return

    os.chmod(target, stat.S_IWRITE)
    target.unlink()


def reset_output_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for child in DATA_DIR.iterdir():
        remove_path(child)


def main() -> None:
    if not URMP_ROOT.exists():
        print(f"URMP folder was not found: {URMP_ROOT}")
        return

    reset_output_dir()

    counts = {label: 0 for label in TARGET_INSTRUMENTS.values()}

    wav_files = list(URMP_ROOT.rglob("*.wav"))
    print(f"Found wav files: {len(wav_files)}")

    for wav_path in wav_files:
        if not is_target_ausep_file(wav_path):
            continue

        instrument = detect_instrument(wav_path.name)
        if instrument is None:
            continue

        if counts[instrument] >= MAX_FILES_PER_INSTRUMENT:
            continue

        instrument_dir = DATA_DIR / instrument
        instrument_dir.mkdir(exist_ok=True)

        new_name = f"{wav_path.parent.name}_{wav_path.name}"
        dest_path = instrument_dir / new_name

        shutil.copy2(wav_path, dest_path)
        counts[instrument] += 1
        print(f"copy: {instrument} -> {dest_path.name}")

    print("\nCopy summary")
    for instrument, count in counts.items():
        print(f"{instrument}: {count} files")


if __name__ == "__main__":
    main()

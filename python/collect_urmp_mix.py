from pathlib import Path
import csv
import re
import shutil

# URMP dataset root
URMP_ROOT = Path(r"C:\URMP")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Output directory for mixed ensemble audio
DATA_DIR = PROJECT_ROOT / "wavlist" / "URMP_Mix"
METADATA_PATH = DATA_DIR / "metadata.csv"

# Target URMP instrument codes
TARGET_INSTRUMENTS = {
    "vn": "violin",
    "vc": "cello",
    "fl": "flute",
    "ob": "oboe",
    "hn": "horn",
}


def is_target_aumix_file(wav_path: Path) -> bool:
    lower_name = wav_path.name.lower()

    # Skip hidden AppleDouble-style files such as ._xxx.wav
    if lower_name.startswith("._"):
        return False

    return "aumix" in lower_name


def detect_instruments(text: str) -> list[str]:
    lower_text = text.lower()
    instruments: list[str] = []

    for short_name, label in TARGET_INSTRUMENTS.items():
        pattern = rf"(^|[^a-z0-9]){re.escape(short_name)}([^a-z0-9]|$)"
        if re.search(pattern, lower_text):
            instruments.append(label)

    return instruments


def main() -> None:
    if not URMP_ROOT.exists():
        print(f"URMP folder was not found: {URMP_ROOT}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    wav_files = list(URMP_ROOT.rglob("*.wav"))
    print(f"Found wav files: {len(wav_files)}")

    for wav_path in wav_files:
        if not is_target_aumix_file(wav_path):
            continue

        instrument_text = f"{wav_path.parent.name}_{wav_path.name}"
        instruments = detect_instruments(instrument_text)
        if not instruments:
            continue

        new_name = f"{wav_path.parent.name}_{wav_path.name}"
        dest_path = DATA_DIR / new_name

        shutil.copy2(wav_path, dest_path)
        rows.append({
            "filename": new_name,
            "instruments": ";".join(instruments),
            "source_path": str(wav_path),
        })
        print(f"copy: {','.join(instruments)} -> {new_name}")

    with METADATA_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "instruments", "source_path"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nCopy summary")
    print(f"mixed files: {len(rows)}")
    print(f"metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()

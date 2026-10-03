from pathlib import Path
import shutil

# URMPを置いた場所
URMP_ROOT = Path(r"C:\URMP")

# このPythonファイルの場所から見た「オーケストラ」フォルダ
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# コピー先
DATA_DIR = PROJECT_ROOT / "wavlist/URMP_data"

# URMP内の楽器略称と、こちらで使うラベル名
TARGET_INSTRUMENTS = {
    "vn": "violin",
    "vc": "cello",
    "fl": "flute",
    "ob": "oboe",
    "hn": "horn",
}

# 各楽器ごとに最大何ファイルコピーするか
MAX_FILES_PER_INSTRUMENT = 30


def detect_instrument(file_name: str):
    """
    ファイル名から楽器を判定する。
    URMPでは violin=vn, cello=vc, flute=fl, oboe=ob, horn=hn のような略称が使われることが多い。
    """
    lower_name = file_name.lower()

    for short_name, label in TARGET_INSTRUMENTS.items():
        # ファイル名の一部に _vn_ のように入っている想定
        if f"_{short_name}_" in lower_name or lower_name.startswith(f"{short_name}_"):
            return label

    return None


def main():
    if not URMP_ROOT.exists():
        print(f"URMPフォルダが見つかりません: {URMP_ROOT}")
        return

    DATA_DIR.mkdir(exist_ok=True)

    counts = {label: 0 for label in TARGET_INSTRUMENTS.values()}

    wav_files = list(URMP_ROOT.rglob("*.wav"))
    print(f"見つかったwavファイル数: {len(wav_files)}")

    for wav_path in wav_files:
        instrument = detect_instrument(wav_path.name)

        if instrument is None:
            continue

        if counts[instrument] >= MAX_FILES_PER_INSTRUMENT:
            continue

        instrument_dir = DATA_DIR / instrument
        instrument_dir.mkdir(exist_ok=True)

        # 同じファイル名が重ならないように、親フォルダ名も付ける
        new_name = f"{wav_path.parent.name}_{wav_path.name}"
        dest_path = instrument_dir / new_name

        shutil.copy2(wav_path, dest_path)
        counts[instrument] += 1

        print(f"copy: {instrument} -> {dest_path.name}")

    print("\nコピー結果")
    for instrument, count in counts.items():
        print(f"{instrument}: {count} files")


if __name__ == "__main__":
    main()
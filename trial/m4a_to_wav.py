import shutil
import sys
from pathlib import Path

print("Python:", sys.executable, file=sys.stderr, flush=True)

try:
    from pydub import AudioSegment
except ModuleNotFoundError as exc:
    print("pydub import failed.", file=sys.stderr, flush=True)
    print("Install it into the Python shown above with:", file=sys.stderr, flush=True)
    print(f'"{sys.executable}" -m pip install pydub', file=sys.stderr, flush=True)
    raise

print("pydub OK")

ffmpeg_path = shutil.which("ffmpeg")
ffprobe_path = shutil.which("ffprobe")
print("ffmpeg:", ffmpeg_path)
print("ffprobe:", ffprobe_path)

if not ffmpeg_path or not ffprobe_path:
    raise FileNotFoundError(
        "ffmpeg and/or ffprobe were not found in PATH. "
        "Install ffmpeg, then reopen VS Code or the terminal."
    )

project_root = Path(__file__).resolve().parent.parent
input_folder = project_root / "datas"
output_folder = project_root / "datas_wav"
output_folder.mkdir(exist_ok=True)

m4a_files = sorted(input_folder.glob("*.m4a"))
if not m4a_files:
    print(f"No .m4a files were found in {input_folder}")

converted_count = 0
skipped_count = 0

for m4a_path in m4a_files:
    wav_path = output_folder / f"{m4a_path.stem}.wav"

    if wav_path.exists():
        m4a_mtime = m4a_path.stat().st_mtime
        wav_mtime = wav_path.stat().st_mtime
        if wav_mtime >= m4a_mtime:
            print(f"Skipped: {m4a_path.name} -> {wav_path.name} is up to date")
            skipped_count += 1
            continue

    audio = AudioSegment.from_file(m4a_path, format="m4a")
    audio.export(wav_path, format="wav")
    print(f"Converted: {m4a_path.name} -> {wav_path.name}")
    converted_count += 1

print(f"Done (converted={converted_count}, skipped={skipped_count})")

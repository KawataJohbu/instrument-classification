"""マイクの楽器識別結果を図形で表示する。--demo でマイクなしの確認も可能。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk

# このスクリプトの場所を基準にするため、別の作業フォルダからも実行できる。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

INSTRUMENTS = (
    ("violin", "ヴァイオリン", "circle", "#63dbad"),
    ("flute", "フルート", "square", "#69c8ff"),
    ("horn", "ホルン", "triangle", "#ffd166"),
    ("cello", "チェロ", "diamond", "#c4a0ff"),
    ("oboe", "オーボエ", "star", "#ff93b8"),
)


def polygon_points(shape: str, x: float, y: float, radius: float) -> list[float]:
    """中心と外接円半径から頂点を生成。三角形は正三角形。"""
    count = {"square": 4, "triangle": 3, "diamond": 4, "star": 10}[shape]
    offset = -math.pi / 4 if shape == "square" else -math.pi / 2
    points = []
    for index in range(count):
        angle = offset + index * 2 * math.pi / count
        size = radius * (0.44 if shape == "star" and index % 2 else 1)
        points.extend((x + math.cos(angle) * size, y + math.sin(angle) * size))
    return points


def audio_worker(args, events: queue.Queue, stop: threading.Event) -> None:
    """録音・学習は別スレッド。Tkinterの操作は一切行わない。"""
    try:
        import orchestra
        import orchestra_live as live

        events.put(("status", "モデルを準備しています…"))
        data_dir = Path(args.data_dir)
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir
        model = live.load_or_train_model(
            data_dir, orchestra.DEFAULT_MAX_SECONDS, orchestra.DEFAULT_SEGMENT_SECONDS,
            retrain=args.retrain,
        )
        while not stop.is_set():
            events.put(("status", f"録音中：{args.window_seconds:g}秒"))
            samples = live.record_mono_samples(args.sample_rate, args.window_seconds, args.device)
            if stop.is_set():
                return
            rms = orchestra.calculate_rms(samples)
            if rms < args.min_rms:
                events.put(("result", ({}, rms, "音量不足：判定をスキップ")))
                continue
            events.put(("status", "音を解析しています…"))
            features = orchestra.extract_features_from_samples(
                Path("live_microphone.wav"), samples, args.sample_rate,
            )
            scores, _ = orchestra.sklearn_file_scores(model, [features])
            events.put(("result", (scores, rms, "識別結果を更新しました")))
    except Exception as error:
        if not stop.is_set():
            events.put(("error", f"{type(error).__name__}: {error}"))


class InstrumentWindow:
    def __init__(self, root: tk.Tk, args):
        self.root, self.args = root, args
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.scores = {}
        self.levels = {key: 0.0 for key, *_ in INSTRUMENTS}
        self.last_result = 0.0
        self.status = "準備中…"
        self.result_text = "まだ判定結果がありません"
        self.rms = None
        self.started = time.monotonic()
        root.title("Orchestra — 楽器の識別アニメーション")
        root.geometry("1050x560")
        root.minsize(760, 480)
        self.canvas = tk.Canvas(root, bg="#101827", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _: self.close())
        if args.demo:
            self.status = "デモ表示：実際の音声は使用していません"
        else:
            threading.Thread(target=audio_worker, args=(args, self.events, self.stop), daemon=True).start()
        self.tick()

    def close(self):
        self.stop.set()
        # 録音待ちを解除する。学習スレッドはdaemonなので終了を妨げない。
        if not self.args.demo and "sounddevice" in sys.modules:
            sys.modules["sounddevice"].stop()
        self.root.destroy()

    def tick(self):
        now = time.monotonic()
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "result":
                self.scores, self.rms, self.result_text = value
                self.last_result = now
            elif kind == "error":
                self.status = f"エラー：{value}"
                self.result_text = "マイク・学習データ・依存ライブラリを確認してください。"
                self.scores = {}
            else:
                self.status = value
        if self.args.demo:
            index = int((now - self.started) / 2) % len(INSTRUMENTS)
            self.scores = {row[0]: (80.0 if i == index else 5.0) for i, row in enumerate(INSTRUMENTS)}
            self.last_result = now
            self.result_text = "2秒ごとに図形が点灯します"
        # 処理が遅れた場合に、古い結果を点灯し続けない。
        fresh = now - self.last_result < max(8.0, self.args.window_seconds * 3)
        for key in self.levels:
            target = float(fresh and self.scores.get(key, 0) >= self.args.threshold)
            self.levels[key] += (target - self.levels[key]) * 0.12
        self.draw(now, fresh)
        self.root.after(33, self.tick)

    def draw(self, now, fresh):
        c = self.canvas
        c.delete("all")
        width, height = c.winfo_width(), c.winfo_height()
        if width < 10:
            return
        c.create_text(32, 35, anchor="w", text="ORCHESTRA", fill="#f0f5ff", font=("", 24, "bold"))
        c.create_text(32, 76, anchor="w", text="楽器の識別結果に合わせて図形が点灯します", fill="#9aabc2", font=("", 12))
        active = [name for key, name, *_ in INSTRUMENTS if fresh and self.scores.get(key, 0) >= self.args.threshold]
        summary = "基準以上：" + "・".join(active) if active else "基準以上の楽器：なし"
        c.create_text(32, 120, anchor="w", text=summary, fill="#e4edf9", font=("", 15, "bold"))
        y = height * 0.48
        spacing = (width - 40) / 5
        for index, (key, name, shape, color) in enumerate(INSTRUMENTS):
            x = 20 + spacing * (index + 0.5)
            level = self.levels[key]
            radius = min(48, spacing * 0.28) * (1 + level * 0.12 * math.sin(now * 5))
            fill = self.blend("#28354a", color, level)
            if level > 0.02:
                ring = radius + 13 + 4 * math.sin(now * 4)
                c.create_oval(x-ring, y-ring, x+ring, y+ring, outline=self.blend("#101827", color, level * 0.5), width=2)
            if shape == "circle":
                c.create_oval(x-radius, y-radius, x+radius, y+radius, fill=fill, outline="")
            else:
                c.create_polygon(polygon_points(shape, x, y, radius), fill=fill, outline="")
            c.create_text(x, y+86, text=name, fill="#e4edf9", font=("", 13, "bold"))
            score = self.scores.get(key)
            text = f"{score:.1f} / 100" if score is not None and fresh else "—"
            detected = fresh and score is not None and score >= self.args.threshold
            c.create_text(x, y+115, text=text, fill=color if detected else "#9aabc2", font=("", 12))
        c.create_text(32, height-112, anchor="w", text=self.status, width=width-64, fill="#ffd166" if self.status.startswith("エラー") else "#9aabc2", font=("", 11))
        age = f" ／ {now-self.last_result:.1f}秒前の結果" if self.last_result else ""
        volume = f" ／ RMS：{self.rms:.1f}" if self.rms is not None else ""
        c.create_text(32, height-68, anchor="w", text=self.result_text+volume+age, width=width-64, fill="#9aabc2", font=("", 10))
        c.create_text(32, height-30, anchor="w", text=f"相対スコアの表示基準：{self.args.threshold:g} / 100（確率・同時演奏の確認ではありません）  |  Esc：終了", fill="#73859e", font=("", 10))

    @staticmethod
    def blend(start: str, end: str, amount: float) -> str:
        return "#" + "".join(f"{round(int(start[i:i+2], 16)*(1-amount)+int(end[i:i+2], 16)*amount):02x}" for i in (1, 3, 5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="マイクを使わず図形の動きを確認")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--min-rms", type=float, default=150.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--device", type=int)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "wavlist" / "URMP_Sep"))
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0 or args.sample_rate <= 0:
        parser.error("録音時間とサンプルレートは正の値を指定してください。")
    if not 0 <= args.threshold <= 100 or not math.isfinite(args.min_rms) or args.min_rms < 0:
        parser.error("thresholdは0〜100、min-rmsは0以上を指定してください。")
    root = tk.Tk()
    InstrumentWindow(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

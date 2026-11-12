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

import graph
from scenes import SCENES
from PIL import Image, ImageOps, ImageTk

# graph.pyで調整した名前付き関数を、楽器ごとに指定する。
GRAPH_SHAPES = {
    "horn": ("cloud1", 130),
    "flute": ("bubbleL", 55),
    "cello": ("leaf_front", 115),
}

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
            started = time.perf_counter()
            samples, sample_rate = live.record_input(args, stop)
            recorded = time.perf_counter()
            if stop.is_set():
                return
            rms = orchestra.calculate_rms(samples)
            if rms < args.min_rms:
                events.put(("result", ({}, rms, "音量不足：判定をスキップ")))
                continue
            events.put(("status", "音を解析しています…"))
            features = orchestra.extract_features_from_samples(
                Path("live_microphone.wav"), samples, sample_rate,
            )
            scores, _ = orchestra.sklearn_file_scores(model, [features])
            if args.timing:
                print(f"Capture: {recorded-started:.3f}s; analysis: {time.perf_counter()-recorded:.3f}s", flush=True)
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
        self.scenes = SCENES
        self.current_scene = 1
        self.instruments = INSTRUMENTS
        self.graph_shapes = GRAPH_SHAPES
        self.polygon_points = polygon_points
        self.graph_masks = {}
        self.graph_photos = []
        self.graph_revision = None
        self.graph_errors = set()
        root.title("Orchestra — 楽器の識別アニメーション")
        root.geometry("1050x560")
        root.minsize(760, 480)
        self.canvas = tk.Canvas(root, bg="#101827", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        # プレビューは表示せず、保存時の再読み込みだけを利用する。
        graph_frame = tk.Frame(root, bg=graph.BACKGROUND)
        # graph.pyの名前付き描画関数を使う。500msごとの再読み込み・
        # エラー時に直前の表示を保つ処理はgraph.LivePreviewに共通化する。
        # 録音スレッドや学習モデルは再起動しない。
        self.graph_preview = graph.LivePreview(root, graph_frame, graph.__file__)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _: self.close())
        for number in self.scenes:
            root.bind(str(number), lambda event, selected=number: self.select_scene(selected))
        if args.demo:
            self.status = "デモ表示：実際の音声は使用していません"
        else:
            threading.Thread(target=audio_worker, args=(args, self.events, self.stop), daemon=True).start()
        self.tick()

    def select_scene(self, number):
        """描画だけを切り替え、録音・識別・スコアの状態は引き継ぐ。"""
        if number in self.scenes:
            self.current_scene = number
            print(f"[scene] {number}", flush=True)

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
                print(self.status, flush=True)
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
        self.scenes[self.current_scene].draw(self, now, fresh)

    def draw_graph_icon(self, key, x, y, radius, color):
        """graphの図形を滑らかに描き、識別状態の色を付けて表示する。"""
        namespace = self.graph_preview.namespace
        if namespace is None:
            return
        revision = id(namespace)
        if revision != self.graph_revision:
            self.graph_masks.clear()
            self.graph_errors.clear()
            self.graph_revision = revision
        name, base_radius = GRAPH_SHAPES[key]
        size = max(16, math.ceil(radius*2+16))
        cache_key = (name, size)
        try:
            if cache_key not in self.graph_masks:
                surface = namespace["SmoothCanvas"](size, size, "white", namespace["ANTIALIAS_SCALE"])
                namespace[name](surface, size/2, size/2, scale=(size-16)/2/base_radius)
                mask = ImageOps.invert(surface.resized_image().convert("L"))
                self.graph_masks[cache_key] = mask
                self.graph_masks[(name, "last")] = mask
            mask = self.graph_masks[cache_key]
        except Exception as error:
            if name not in self.graph_errors:
                print(f"[graph] {name}: {error}. Keeping the previous shape.", flush=True)
                self.graph_errors.add(name)
            mask = self.graph_masks.get((name, "last"))
            if mask is None:
                return
        image = Image.new("RGBA", mask.size, color)
        image.putalpha(mask)
        photo = ImageTk.PhotoImage(image, master=self.canvas)
        self.graph_photos.append(photo)
        self.canvas.create_image(x, y, image=photo)

    @staticmethod
    def blend(start: str, end: str, amount: float) -> str:
        return "#" + "".join(f"{round(int(start[i:i+2], 16)*(1-amount)+int(end[i:i+2], 16)*amount):02x}" for i in (1, 3, 5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    import orchestra_live as live
    live.add_input_options(parser)
    parser.add_argument("--demo", action="store_true", help="マイクを使わず図形の動きを確認")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--min-rms", type=float, default=150.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--device", type=int)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "wavlist" / "URMP_Sep"))
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    if args.list_loopback_devices:
        live.print_loopback_devices()
        return
    if args.loopback_device is not None and not args.loopback:
        parser.error("--loopback-device requires --loopback")
    if args.loopback and args.device is not None:
        parser.error("Use --loopback-device for loopback, not --device")
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0 or args.sample_rate <= 0:
        parser.error("録音時間とサンプルレートは正の値を指定してください。")
    if not 0 <= args.threshold <= 100 or not math.isfinite(args.min_rms) or args.min_rms < 0:
        parser.error("thresholdは0〜100、min-rmsは0以上を指定してください。")
    root = tk.Tk()
    InstrumentWindow(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

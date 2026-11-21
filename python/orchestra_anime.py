"""マイクの楽器識別結果を図形で表示する。--demo でマイクなしの確認も可能。"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import math
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk

# このスクリプトの場所を基準にするため、別の作業フォルダからも実行できる。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))

import graph
from scenes import SCENES
from scenes.common import BACKGROUND
from leaf_loop import LeafLoop
from display_reload import load_display_modules, restore_modules
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
        self.graph_photo_cache = OrderedDict()
        self.graph_items = {}
        self.drawn_scene = None
        self.scene1_overlay = None
        self.frame_stats = [time.perf_counter(), 0, 0.0]
        root.title("Orchestra — 楽器の識別アニメーション")
        root.geometry(f"{args.width}x{args.height}")
        root.minsize(200, 200)
        self.canvas = tk.Canvas(root, bg=BACKGROUND, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.leaf_loop = LeafLoop()
        # 再生中の初回変換による停止を避けるため、起動時に全コマを用意する。
        self.leaf_loop.prepare(.045*min(args.width, args.height), self.canvas)
        # プレビューは表示せず、保存時の再読み込みだけを利用する。
        graph_frame = tk.Frame(root, bg=graph.BACKGROUND)
        # graph.pyの名前付き描画関数を使う。500msごとの再読み込み・
        # エラー時に直前の表示を保つ処理はgraph.LivePreviewに共通化する。
        # 録音スレッドや学習モデルは再起動しない。
        self.graph_preview = graph.LivePreview(root, graph_frame, graph.__file__)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _: self.close())
        root.bind("<KeyPress-r>", lambda _: self.reload_display())
        root.bind("<KeyPress-R>", lambda _: self.reload_display())
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

    def reload_display(self):
        """Rで表示コードとPNGを読み直す。録音スレッドと識別結果は維持する。"""
        previous_modules = None
        old_class = self.__class__
        old_state = self.__dict__.copy()
        preview_state = (self.graph_preview.namespace, self.graph_preview.last_source)
        try:
            modules, previous_modules = load_display_modules()
            new_graph = modules["graph"]
            # 描画と素材を検証してから、現在の表示へ反映する。
            new_graph.render_image(320, 240)
            player = modules["leaf_loop"].LeafLoop()
            player.prepare(.045*min(self.canvas.winfo_width(), self.canvas.winfo_height()), self.canvas)
            app = modules["_orchestra_anime_live"]
            self.__class__ = app.InstrumentWindow
            self.scenes = modules["scenes"].SCENES
            self.instruments = app.INSTRUMENTS
            self.graph_shapes = app.GRAPH_SHAPES
            self.polygon_points = app.polygon_points
            self.leaf_loop = player
            self.graph_preview.namespace = vars(new_graph)
            self.graph_preview.last_source = Path(new_graph.__file__).read_bytes()
            self.scene_states = {}
            self.graph_masks = {}
            self.graph_photos = []
            self.graph_photo_cache = OrderedDict()
            self.graph_items = {}
            self.graph_errors = set()
            self.graph_revision = None
            self.scene1_overlay = None
            self.drawn_scene = None
            now = time.monotonic()
            fresh = now-self.last_result < max(8., self.args.window_seconds*3)
            self.draw(now, fresh)
            print("[reload] Updated scenes, graph, leaf PNGs and display code.", flush=True)
        except Exception as error:
            if previous_modules is not None:
                restore_modules(previous_modules)
            self.__class__ = old_class
            self.__dict__.clear()
            self.__dict__.update(old_state)
            self.graph_preview.namespace, self.graph_preview.last_source = preview_state
            self.drawn_scene = None
            print(f"[reload] Update failed; keeping previous code: {type(error).__name__}: {error}", flush=True)

    def close(self):
        self.stop.set()
        # 録音待ちを解除する。学習スレッドはdaemonなので終了を妨げない。
        if not self.args.demo and "sounddevice" in sys.modules:
            sys.modules["sounddevice"].stop()
        self.root.destroy()

    def tick(self):
        frame_started = time.perf_counter()
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
        elapsed = time.perf_counter()-frame_started
        self.frame_stats[1] += 1
        self.frame_stats[2] += elapsed
        span = time.perf_counter()-self.frame_stats[0]
        if self.args.timing and span >= 2:
            print(f"Render: {self.frame_stats[1]/span:.1f} fps; frame work: {1000*self.frame_stats[2]/self.frame_stats[1]:.2f} ms", flush=True)
            self.frame_stats = [time.perf_counter(), 0, 0.0]
        # 描画時間を差し引いて待ち、描画時間+固定待機によるfps低下を避ける。
        self.root.after(max(1, round(1000/60-elapsed*1000)), self.tick)

    def draw(self, now, fresh):
        if self.drawn_scene != self.current_scene:
            self.canvas.delete("all")
            self.graph_items.clear()
            self.scene1_overlay = None
            self.drawn_scene = self.current_scene
        self.scenes[self.current_scene].draw(self, now, fresh)

    def draw_graph_icon(self, key, x, y, radius, color):
        """graphの図形を滑らかに描き、識別状態の色を付けて表示する。"""
        name, base_radius = GRAPH_SHAPES[key]
        self.draw_graph_shape(name, x, y, radius, color, base_radius)

    def draw_leaf_loop(self, x, y, radius, elapsed, phase, item_key, rotation=0.):
        photo = self.leaf_loop.photo(radius, elapsed, phase, self.canvas, rotation=rotation)
        self.place_graph_photo(photo, x, y, item_key)

    def draw_graph_shape(self, name, x, y, radius, color="black", base_radius=130, item_key=None):
        """シーンからgraphの名前付き関数を直接選び、描画する。"""
        namespace = self.graph_preview.namespace
        if namespace is None:
            return
        revision = id(namespace)
        if revision != self.graph_revision:
            self.graph_masks.clear()
            self.graph_errors.clear()
            self.graph_photo_cache.clear()
            self.graph_revision = revision
        extent = namespace.get("SHAPE_EXTENT_FACTORS", {}).get(name, 1.0)
        size = max(16, math.ceil(radius*2*extent+16))
        cache_key = (name, size, base_radius)
        photo_key = (name, size, base_radius, color)
        photo = self.graph_photo_cache.get(photo_key)
        if photo is not None:
            self.graph_photo_cache.move_to_end(photo_key)
            self.place_graph_photo(photo, x, y, item_key)
            return
        try:
            if cache_key not in self.graph_masks:
                surface = namespace["SmoothCanvas"](size, size, "white", namespace["ANTIALIAS_SCALE"])
                namespace[name](surface, size/2, size/2, scale=radius/base_radius)
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
        self.graph_photo_cache[photo_key] = photo
        # サイズ変更や色変化を繰り返しても無制限にメモリを使わない。
        while len(self.graph_photo_cache) > 128:
            self.graph_photo_cache.popitem(last=False)
        while len(self.graph_masks) > 128:
            self.graph_masks.pop(next(iter(self.graph_masks)))
        self.place_graph_photo(photo, x, y, item_key)

    def place_graph_photo(self, photo, x, y, item_key=None):
        if item_key is None:
            self.graph_photos.append(photo)
            self.canvas.create_image(x, y, image=photo)
            return
        previous = self.graph_items.get(item_key)
        if previous is None:
            item = self.canvas.create_image(x, y, image=photo)
        else:
            item, old_photo = previous
            self.canvas.coords(item, x, y)
            if old_photo is not photo:
                self.canvas.itemconfigure(item, image=photo)
        # Canvasが使っている画像の参照はLRUから外れても保持する。
        self.graph_items[item_key] = (item, photo)

    def finish_graph_items(self, active_keys):
        for key in set(self.graph_items)-active_keys:
            item, _ = self.graph_items.pop(key)
            self.canvas.delete(item)

    @staticmethod
    def blend(start: str, end: str, amount: float) -> str:
        return "#" + "".join(f"{round(int(start[i:i+2], 16)*(1-amount)+int(end[i:i+2], 16)*amount):02x}" for i in (1, 3, 5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    import orchestra_live as live
    live.add_input_options(parser)
    parser.add_argument("--demo", action="store_true", help="マイクを使わず図形の動きを確認")
    parser.add_argument("--width", type=int, default=1050, help="Window width in pixels (e.g. 1920).")
    parser.add_argument("--height", type=int, default=560, help="Window height in pixels (e.g. 1920).")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--min-rms", type=float, default=150.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--device", type=int)
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "wavlist" / "URMP_Sep"))
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    if args.width < 200 or args.height < 200:
        parser.error("Window width and height must be at least 200.")
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

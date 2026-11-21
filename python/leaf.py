"""葉の奥行き方向の回転を確認する。実行例：py python/leaf.py"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import math
from pathlib import Path
import time
import tkinter as tk

from PIL import Image, ImageOps, ImageTk

import graph

TIP_CURL = 0.12  # 正面での先端の縮み具合。0なら従来の形。
ROOT_CURL = 0.0  # 根元と茎は曲げない。
DEPTH_CURVE = 0.10  # 横向きのときに見える葉の反り。


def tilt_at(elapsed: float, cycle_seconds: float = 8.0) -> float:
    """1周の時間から連続した傾きを求める。画面内では回転させない。"""
    return (elapsed / cycle_seconds * 360.0) % 360.0


def render_leaf(size: int, tilt: float, namespace=None, transparent=False):
    """graphの葉を任意の傾きで描く。ウィンドウなしでも利用できる。"""
    ns = vars(graph) if namespace is None else namespace
    surface = ns["SmoothCanvas"](size, size, "white", ns["ANTIALIAS_SCALE"])
    facing = math.cos(math.radians(tilt))**2
    curl = TIP_CURL * facing
    ns["draw_leaf"](surface, size/2, size/2, scale=(size-16)/230,
                    rotation=0, tilt=tilt, tip_curl=curl, root_curl=ROOT_CURL*facing,
                    depth_curve=DEPTH_CURVE)
    image = surface.resized_image()
    if transparent:
        # 白背景をアルファに変換。縁の半透明画素も黒にして白い縁を防ぐ。
        result = Image.new("RGBA", image.size, (0, 0, 0, 0))
        result.putalpha(ImageOps.invert(image.convert("L")))
        return result
    return image


def export_frames(output, frames=60, size=256, cycle_seconds=8.0):
    """一周を等分して保存する。360度は0度と重複するため含めない。"""
    if frames < 1 or size < 24 or not math.isfinite(cycle_seconds) or cycle_seconds <= 0:
        raise ValueError("frames>=1, size>=24 and a positive finite cycle duration are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    rows = []
    for index in range(frames):
        elapsed = index*cycle_seconds/frames
        angle = index*360.0/frames
        filename = f"leaf_{index:03d}.png"
        path = output/filename
        render_leaf(size, angle, transparent=True).save(path)
        paths.append(path)
        rows.append((index, filename, angle, elapsed, cycle_seconds/frames))
    with (output/"frames.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("frame", "filename", "angle_degrees", "time_seconds", "duration_seconds"))
        writer.writerows(rows)
    return paths


class LeafAnimation:
    def __init__(self, root, cycle_seconds=8.0):
        self.root = root
        self.cycle_seconds = cycle_seconds
        self.started = time.perf_counter()
        self.photos = OrderedDict()
        self.revision = None
        self.current_photos = [None]
        self.last_keys = [None]
        self.failed_revision = None
        self.canvas = tk.Canvas(root, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.items = [self.canvas.create_image(0, 0)]
        # graph.pyを保存したら読み直す。確認用の別ウィンドウは開かない。
        hidden_frame = tk.Frame(root, bg="white")
        self.preview = graph.LivePreview(root, hidden_frame, graph.__file__)
        root.bind("<Escape>", lambda event: root.destroy())
        self.tick()

    def tick(self):
        started = time.perf_counter()
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        namespace = self.preview.namespace
        if width > 1 and height > 1 and namespace is not None:
            revision = id(namespace)
            if revision != self.revision:
                self.photos.clear()
                self.last_keys = [None]
                self.failed_revision = None
                self.revision = revision
            size = max(24, round(min(width, height)*.70))
            tilt = tilt_at(started-self.started, self.cycle_seconds)
            # 元の方向に回転する葉を1つ表示する。
            for index, direction in enumerate((1,)):
                angle = round(direction*tilt) % 360
                key = (size, angle)
                if key != self.last_keys[index] and self.failed_revision != revision:
                    try:
                        photo = self.photos.get(key)
                        if photo is None:
                            photo = ImageTk.PhotoImage(render_leaf(size, angle, namespace), master=self.canvas)
                            self.photos[key] = photo
                        self.photos.move_to_end(key)
                        while (len(self.photos) > 128 or
                               sum(k[0]*k[0]*4 for k in self.photos) > 64*1024*1024):
                            self.photos.popitem(last=False)
                        self.canvas.itemconfigure(self.items[index], image=photo)
                        self.current_photos[index] = photo
                        self.last_keys[index] = key
                    except Exception as error:
                        print(f"[leaf] {error}", flush=True)
                        self.failed_revision = revision
                        self.last_keys[index] = key
                self.canvas.coords(self.items[index], width/2, height/2)
        elapsed = time.perf_counter()-started
        self.root.after(max(1, round(1000/60-elapsed*1000)), self.tick)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-seconds", type=float, default=8.0,
                        help="Seconds per complete rotation (default: 8).")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--export", action="store_true", help="Export transparent PNG frames without opening a window.")
    parser.add_argument("--frames", type=int, default=60, help="Number of exported frames (default: 60).")
    parser.add_argument("--size", type=int, default=256, help="Exported PNG width and height (default: 256).")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent/"assets"/"leaf",
                        help="PNG export directory.")
    args = parser.parse_args()
    if not math.isfinite(args.cycle_seconds) or args.cycle_seconds <= 0:
        parser.error("cycle-seconds must be positive and finite")
    if args.width < 200 or args.height < 200:
        parser.error("width and height must be at least 200")
    if args.frames < 1 or args.size < 24:
        parser.error("frames must be at least 1 and size must be at least 24")
    if args.export:
        paths = export_frames(args.output, args.frames, args.size, args.cycle_seconds)
        print(f"Exported {len(paths)} transparent PNG frames ({args.size}x{args.size}) to {args.output}")
        return
    root = tk.Tk()
    root.title("Leaf")
    root.geometry(f"{args.width}x{args.height}")
    root.minsize(200, 200)
    animation = LeafAnimation(root, args.cycle_seconds)
    root.mainloop()


if __name__ == "__main__":
    main()

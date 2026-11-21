"""葉の奥行き方向の回転を確認する。実行例：py python/leaf.py"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import math
import time
import tkinter as tk

from PIL import ImageTk

import graph

TIP_CURL = 0.12  # 正面での先端の縮み具合。0なら従来の形。
ROOT_CURL = 0.0  # 根元と茎は曲げない。
DEPTH_CURVE = 0.10  # 横向きのときに見える葉の反り。


def tilt_at(elapsed: float, cycle_seconds: float = 8.0) -> float:
    """1周の時間から連続した傾きを求める。画面内では回転させない。"""
    return (elapsed / cycle_seconds * 360.0) % 360.0


def render_leaf(size: int, tilt: float, namespace=None):
    """graphの葉を任意の傾きで描く。ウィンドウなしでも利用できる。"""
    ns = vars(graph) if namespace is None else namespace
    surface = ns["SmoothCanvas"](size, size, "white", ns["ANTIALIAS_SCALE"])
    facing = math.cos(math.radians(tilt))**2
    curl = TIP_CURL * facing
    ns["draw_leaf"](surface, size/2, size/2, scale=(size-16)/230,
                    rotation=0, tilt=tilt, tip_curl=curl, root_curl=ROOT_CURL*facing,
                    depth_curve=DEPTH_CURVE)
    return surface.resized_image()


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
    args = parser.parse_args()
    if not math.isfinite(args.cycle_seconds) or args.cycle_seconds <= 0:
        parser.error("cycle-seconds must be positive and finite")
    if args.width < 200 or args.height < 200:
        parser.error("width and height must be at least 200")
    root = tk.Tk()
    root.title("Leaf")
    root.geometry(f"{args.width}x{args.height}")
    root.minsize(200, 200)
    animation = LeafAnimation(root, args.cycle_seconds)
    root.mainloop()


if __name__ == "__main__":
    main()

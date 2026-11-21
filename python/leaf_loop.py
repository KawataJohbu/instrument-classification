"""透過PNGを一度読み込み、複数の葉で共有するループプレイヤー。"""

from bisect import bisect_right
from collections import OrderedDict
import csv
import math
from pathlib import Path

from PIL import Image, ImageTk


class LeafLoop:
    def __init__(self, folder=None):
        folder = Path(folder) if folder else Path(__file__).resolve().parent/"assets"/"leaf"
        self.frames = []
        self.ends = []
        self.duration = 0.
        durations = []
        try:
            with (folder/"frames.csv").open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            for row in rows:
                path = (folder/row["filename"]).resolve()
                if path.parent != folder.resolve():
                    raise ValueError("Invalid frame path")
                duration = float(row["duration_seconds"])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("Invalid frame duration")
                with Image.open(path) as image:
                    frame = image.convert("RGBA")
                    if self.frames and frame.size != self.frames[0].size:
                        raise ValueError("Leaf frames must have the same dimensions")
                    self.frames.append(frame)
                durations.append(duration)
                self.duration = math.fsum(durations)
                self.ends.append(self.duration)
        except (OSError, KeyError, ValueError) as error:
            raise RuntimeError("Cannot load leaf PNGs. Run: python python/leaf.py --export --frames 60") from error
        if not self.frames:
            raise ValueError("No leaf frames in frames.csv")
        self.photo_sets = OrderedDict()
        self.rotated_photos = OrderedDict()

    def frame_index(self, elapsed, phase=0.):
        moment = (elapsed+phase*self.duration) % self.duration
        return min(bisect_right(self.ends, moment), len(self.frames)-1)

    def prepare(self, radius, master):
        # Match the scale used by graph.draw_leaf; keep the whole PNG canvas.
        source_size = self.frames[0].width
        size = max(16, round(source_size*radius*2/(source_size-16)))
        key = (size, id(master))
        if key not in self.photo_sets:
            photos = tuple(ImageTk.PhotoImage(
                frame.resize((size, size), Image.Resampling.LANCZOS), master=master,
            ) for frame in self.frames)
            self.photo_sets[key] = photos
            # Only two window sizes are retained, to keep resize memory bounded.
            while len(self.photo_sets) > 2:
                self.photo_sets.popitem(last=False)
        self.photo_sets.move_to_end(key)
        return self.photo_sets[key]

    def photo(self, radius, elapsed, phase, master, rotation=0.):
        index = self.frame_index(elapsed, phase)
        # 1度刻みの画像を共有し、毎フレーム同じ角度を作り直さない。
        angle = round(rotation) % 360
        if angle == 0:
            return self.prepare(radius, master)[index]
        source_size = self.frames[0].width
        size = max(16, round(source_size*radius*2/(source_size-16)))
        key = (size, index, angle, id(master))
        if key not in self.rotated_photos:
            image = self.frames[index].resize((size, size), Image.Resampling.LANCZOS)
            # Pillowの正の角度は反時計回り。回転後も画像中心に配置し、
            # 外側へはみ出す葉先や茎が切れないよう透明領域を広げる。
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC,
                                 expand=True, fillcolor=(0, 0, 0, 0))
            self.rotated_photos[key] = ImageTk.PhotoImage(image, master=master)
            while (len(self.rotated_photos) > 256 or
                   sum(k[0]*k[0]*9 for k in self.rotated_photos) > 32*1024*1024):
                self.rotated_photos.popitem(last=False)
        self.rotated_photos.move_to_end(key)
        return self.rotated_photos[key]

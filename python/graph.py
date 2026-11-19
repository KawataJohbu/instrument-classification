r"""雲・シャボン玉・葉っぱの描画サンプル。実行：py .\python\graph.py"""

import math
from pathlib import Path
import tkinter as tk
import traceback

from PIL import Image, ImageDraw, ImageTk


BACKGROUND = "#ffffff"
ANTIALIAS_SCALE = 4  # 大きく描いて縮小する倍率。数値が大きいほど処理量も増える。
# 拡大後の雲を切らずに画像化するため、呼び出し側で確保する描画範囲。
SHAPE_EXTENT_FACTORS = {"cloud1": 1.55, "cloud2": 1.4, "cloud3": 2.2}


class SmoothCanvas:
    """今までのCanvas風の書き方を、高解像度のPillow画像に描画する。"""

    def __init__(self, width, height, background, factor=ANTIALIAS_SCALE):
        self.width, self.height = max(1, width), max(1, height)
        self.factor = factor
        self.background = background
        self.image = Image.new("RGB", (self.width*factor, self.height*factor), background)
        self.pen = ImageDraw.Draw(self.image)

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height

    def delete(self, tag):
        self.pen.rectangle((0, 0, self.image.width, self.image.height), fill=self.background)

    def coordinates(self, values):
        return [value*self.factor for value in values]

    def create_oval(self, *coords, fill="", outline="", width=1):
        self.pen.ellipse(self.coordinates(coords), fill=fill or None,
                         outline=outline or None, width=max(1, round(width*self.factor)))

    def create_arc(self, *coords, start=0, extent=90, style="arc", outline="black", width=1):
        # Tkinterは反時計回り、Pillowは時計回りなので角度を変換する。
        self.pen.arc(self.coordinates(coords), start=-(start+extent), end=-start,
                     fill=outline, width=max(1, round(width*self.factor)))

    def create_polygon(self, *coords, fill="", outline="", width=1):
        points = list(zip(self.coordinates(coords)[::2], self.coordinates(coords)[1::2]))
        if fill:
            self.pen.polygon(points, fill=fill)
        if outline:
            self.pen.line(points+[points[0]], fill=outline,
                          width=max(1, round(width*self.factor)), joint="curve")

    def create_line(self, *coords, fill="black", width=1):
        self.pen.line(self.coordinates(coords), fill=fill,
                      width=max(1, round(width*self.factor)), joint="curve")

    def resized_image(self):
        return self.image.resize((self.width, self.height), Image.Resampling.LANCZOS)


def render_image(width, height):
    """ウィンドウを開かずに画像として描画することもできる。"""
    surface = SmoothCanvas(width, height, BACKGROUND, ANTIALIAS_SCALE)
    draw_scene(surface)
    return surface.resized_image()


def display_scene(canvas):
    image = render_image(canvas.winfo_width(), canvas.winfo_height())
    photo = ImageTk.PhotoImage(image, master=canvas)
    canvas.create_image(0, 0, anchor="nw", image=photo)
    canvas.preview_image = photo  # 参照を保持して、表示画像が消えるのを防ぐ。


def draw_cloud(canvas, x, y, scale=1.0, variant=0, stretch_x=1.0, stretch_y=1.0):
    """楕円の組み合わせを変えた4種類の雲。"""
    variants = [
        # 薄く長く流れる雲。上面の丸みを浅くして出っ張りを抑える。
        [(-136, 8, -61, 45), (-113, -3, -20, 41),
         (-57, -12, 42, 43),
         (30, -8, 103, 42), (79, 12, 139, 45),
         (-103, 13, 110, 51)],
        # 大小のふくらみがゆったり並ぶ、幅広の雲。
        [(-124, -5, -43, 58), (-91, -47, -5, 51),
         (-41, -72, 59, 54), (24, -31, 106, 58),
         (72, 7, 130, 59), (-89, 11, 95, 66)],
        # 縦に積み上がる入道雲。上ほど小さなふくらみにする。
        [(-37, -100, 30, -37), (-61, -67, 35, 19),
         (-11, -42, 67, 32), (-81, -17, 19, 68),
         (-16, 0, 83, 76), (-96, 26, -18, 84),
         (33, 29, 101, 85), (-65, 30, 65, 91)],
        # 大きな丸い塊が左右に並ぶ、ふっくらした雲。
        [(-113, -10, -30, 66), (-80, -70, 20, 49),
         (-15, -56, 90, 61), (45, -5, 116, 65),
         (-60, 3, 70, 75)],
    ]
    parts = variants[variant]
    for left, top, right, bottom in parts:
        canvas.create_oval(
            x + left * scale * stretch_x, y + top * scale * stretch_y,
            x + right * scale * stretch_x, y + bottom * scale * stretch_y,
            fill="#000000", outline="",
        )


def draw_bubble(canvas, x, y, radius=80, variant=0):
    """採用済みの形と、反射・輪郭を変えた2つの候補。"""
    bounds = (x - radius, y - radius, x + radius, y + radius)
    if variant == 1:
        # 細い輪郭に、左上の短い反射と右下の小さな反射。
        canvas.create_oval(*bounds, fill="white", outline="black", width=radius*.035)
        inset = radius*.16
        canvas.create_arc(x-radius+inset, y-radius+inset,
                          x+radius-inset, y+radius-inset,
                          start=100, extent=48, style="arc", outline="black", width=radius*.08)
        canvas.create_oval(x+radius*.48, y+radius*.37,
                           x+radius*.60, y+radius*.49, fill="black", outline="")
        return
    if variant == 2:
        # 太い輪郭に、長い反射と短い反射を対角に配置。右回りに45°回転。
        canvas.create_oval(*bounds, fill="white", outline="black", width=radius*.09)
        for inset, start, extent, weight in ((.19, 70, 100, .055), (.24, 280, 35, .10)):
            canvas.create_arc(x-radius+radius*inset, y-radius+radius*inset,
                              x+radius-radius*inset, y+radius-radius*inset,
                              start=start-45, extent=extent, style="arc",
                              outline="black", width=radius*weight)
        return
    canvas.create_oval(*bounds, fill="white", outline="black", width=radius*.065)
    for inset, start, extent, weight in ((.13, 195, 115, .035), (.20, 42.5, 30, .09)):
        canvas.create_arc(x-radius+radius*inset, y-radius+radius*inset,
                          x+radius-radius*inset, y+radius-radius*inset,
                          start=start, extent=extent, style="arc",
                          outline="black", width=radius*weight)


def draw_leaf(canvas, x, y, scale=1.0, rotation=0, tilt=0):
    """rotationは画面内の回転、tiltは葉の長軸まわりの傾き（90°で真横）。"""
    # 葉の根元から先端への方向と、それに直角な方向。
    base_x, base_y = x - 65*scale, y + 65*scale
    length = 180*scale
    direction = (math.cos(-math.pi/4), math.sin(-math.pi/4))
    normal = (-direction[1], direction[0])
    angle = math.radians(rotation)
    # 葉を長軸まわりに回したときの投影。横から見るほど葉幅が狭くなる。
    # 完全な真横でも消えないよう、わずかな厚みを持たせる。
    projected_width = max(.045, abs(math.cos(math.radians(tilt))))

    def point(t, sideways=0):
        sideways *= projected_width
        px = base_x + direction[0]*length*t + normal[0]*sideways - x
        py = base_y + direction[1]*length*t + normal[1]*sideways - y
        return (x + px*math.cos(angle) - py*math.sin(angle),
                y + px*math.sin(angle) + py*math.cos(angle))

    outline = []
    # sin(pi*t) は両端で0、中央で1。葉の中央だけを膨らませる。
    for side in (1, -1):
        steps = range(31) if side == 1 else range(30, -1, -1)
        for i in steps:
            t = i/30
            outline.extend(point(t, side * 43*scale * math.sin(math.pi*t)))
    canvas.create_polygon(*outline, fill="black", outline="")
    canvas.create_line(*point(-.16), *point(.10), fill="black", width=3*scale)


def cloud1(canvas, x, y, scale=1.0):
    """上段の雲。横1.5倍、縦1.15倍。"""
    draw_cloud(canvas, x, y, scale, variant=1, stretch_x=1.5, stretch_y=1.15)


def cloud2(canvas, x, y, scale=1.0):
    """中段の雲。横1.5倍、縦1.15倍。"""
    draw_cloud(canvas, x, y, scale, variant=3, stretch_x=1.5, stretch_y=1.15)


def cloud3(canvas, x, y, scale=1.0):
    """下段の雲。縦横ともに2倍。"""
    draw_cloud(canvas, x, y, scale, variant=0, stretch_x=2.0, stretch_y=2.0)


def bubbleS(canvas, x, y, scale=1.0):
    """小サイズ。"""
    draw_bubble(canvas, x, y, radius=25*scale)


def bubbleM(canvas, x, y, scale=1.0):
    """中サイズ。"""
    draw_bubble(canvas, x, y, radius=40*scale)


def bubbleL(canvas, x, y, scale=1.0):
    """大サイズ。"""
    draw_bubble(canvas, x, y, radius=55*scale)


def bubbleXL(canvas, x, y, scale=1.0):
    """特大サイズ。"""
    draw_bubble(canvas, x, y, radius=75*scale)


def leaf_front(canvas, x, y, scale=1.0):
    """正面から見た葉。"""
    draw_leaf(canvas, x, y, scale, tilt=0)


def leaf_diagonal(canvas, x, y, scale=1.0):
    """斜めから見た葉。"""
    draw_leaf(canvas, x, y, scale, tilt=58)


def leaf_side(canvas, x, y, scale=1.0):
    """横から見た葉。"""
    draw_leaf(canvas, x, y, scale, tilt=87)


def leaf_back(canvas, x, y, scale=1.0):
    """裏返りかけた葉。"""
    draw_leaf(canvas, x, y, scale, tilt=140)


def draw_scene(canvas):
    """左列に雲、中央にシャボン玉4サイズ、右列に葉っぱの4コマ。"""
    canvas.delete("all")
    width, height = canvas.winfo_width(), canvas.winfo_height()
    # 大きくした雲が切れないよう、左列を広く確保する。
    cloud_scale = min(1.0, width*.24/278, height/600)
    for row, draw in enumerate((cloud1, cloud2, cloud3)):
        draw(canvas, width*.26, height*(row+.5)/3, cloud_scale)
    bubble_scale = min(1.0, width/960, height/680)
    for row, draw in enumerate((bubbleS, bubbleM, bubbleL, bubbleXL)):
        draw(canvas, width*.65, height*(row+.5)/4, bubble_scale)
    leaf_scale = min(0.60, width/1600, height/900)
    # 画面内の向きと横位置は固定し、奥行き方向の傾きだけを変える。
    for row, draw in enumerate((leaf_front, leaf_diagonal, leaf_side, leaf_back)):
        draw(canvas, width*.88, height*(row+.5)/4, leaf_scale)


class LivePreview:
    """500msごとに保存を確認し、成功した描画だけを画面に反映する。"""

    def __init__(self, root, frame, source_path):
        self.root = root
        self.frame = frame
        self.source_path = Path(source_path).resolve()
        self.last_source = None
        self.namespace = None
        self.canvas = None
        self.check_source()
        frame.bind("<Configure>", self.on_resize)

    def render(self, namespace):
        # 別のCanvasに描き切ってから交換する。失敗しても古い図形を残せる。
        candidate = tk.Canvas(self.frame, bg=namespace["BACKGROUND"], highlightthickness=0)
        try:
            candidate.place(x=0, y=0, relwidth=1, relheight=1)
            candidate.update_idletasks()
            namespace["display_scene"](candidate)
        except Exception:
            candidate.destroy()
            raise
        previous = self.canvas
        self.canvas = candidate
        if previous is not None:
            previous.destroy()

    def check_source(self):
        try:
            source = self.source_path.read_bytes()
            if source != self.last_source:
                # 同じエラーを毎回表示せず、次の保存時に再試行する。
                self.last_source = source
                namespace = {
                    "__name__": "graph_preview",  # main()を再実行しない
                    "__file__": str(self.source_path),
                }
                exec(compile(source, str(self.source_path), "exec"), namespace)
                self.render(namespace)
                self.namespace = namespace
                print("[graph] Preview updated.", flush=True)
        except Exception:
            print("[graph] Update failed; keeping the last successful drawing.", flush=True)
            traceback.print_exc()
        finally:
            self.root.after(500, self.check_source)

    def on_resize(self, event):
        if self.namespace is not None:
            try:
                self.render(self.namespace)
            except Exception:
                traceback.print_exc()


def main():
    root = tk.Tk()
    root.title("図形の描画サンプル")
    root.geometry("960x520")
    # root.geometry("1920x1920")
    root.minsize(760, 440)
    frame = tk.Frame(root, bg=BACKGROUND)
    frame.pack(fill="both", expand=True)
    preview = LivePreview(root, frame, __file__)
    root.bind("<Escape>", lambda event: root.destroy())
    root.mainloop()


if __name__ == "__main__":
    main()

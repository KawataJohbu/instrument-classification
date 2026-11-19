"""全sceneで共有する白背景と、画面サイズに対する相対配置。"""

import math

BACKGROUND = "#ffffff"
REFERENCE_SIZE = (1920, 1920)


def relative_point(width, height, x_ratio, y_ratio):
    """中央は(0.5, 0.5)。縦横比が変わっても位置が追従する。"""
    return width*x_ratio, height*y_ratio


def draw_instruments(view, now, fresh, keys=None, use_graph_shapes=True):
    c = view.canvas
    c.configure(bg=BACKGROUND)
    c.delete("all")
    view.graph_photos.clear()
    width, height = c.winfo_width(), c.winfo_height()
    if width < 2 or height < 2:
        return
    by_key = {row[0]: row for row in view.instruments}
    instruments = list(view.instruments) if keys is None else [by_key[key] for key in keys]
    count = len(instruments)
    margin = .04
    spacing = (1-2*margin)/count
    font_size = max(1, round(min(height*.027, width/count*.10)))
    stroke = max(1, min(width, height)/560)
    for index, (key, name, shape, color) in enumerate(instruments):
        x, y = relative_point(width, height, margin+spacing*(index+.5), .40)
        level = view.levels[key]
        radius = min(width*spacing*.28, height*.09)*(1+level*.12*math.sin(now*5))
        fill = view.blend("#c3c8ce", color, level)
        if level > .02:
            ring = radius+height*(.025+.007*math.sin(now*4))
            c.create_oval(x-ring, y-ring, x+ring, y+ring,
                          outline=view.blend(BACKGROUND, color, level*.5), width=2*stroke)
        if use_graph_shapes and key in view.graph_shapes:
            view.draw_graph_icon(key, x, y, radius, fill)
        elif shape == "circle":
            c.create_oval(x-radius, y-radius, x+radius, y+radius, fill=fill, outline="")
        else:
            c.create_polygon(view.polygon_points(shape, x, y, radius), fill=fill, outline="")
        c.create_text(x, y+height*.16, text=name, fill="#20252b",
                      font=("", -font_size, "bold"))
        score = view.scores.get(key)
        text = f"{score:.1f} / 100" if score is not None and fresh else "—"
        c.create_text(x, y+height*.215, text=text, fill="#49515b",
                      font=("", -max(1, round(font_size*.92))))

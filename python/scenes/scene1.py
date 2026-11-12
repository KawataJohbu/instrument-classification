"""Scene 1: the current instrument visualization."""

import math


def draw(view, now, fresh, use_graph_shapes=True):
    c = view.canvas
    c.delete("all")
    view.graph_photos.clear()
    width, height = c.winfo_width(), c.winfo_height()
    if width < 10:
        return
    y = height * 0.40
    spacing = (width - 40) / 5
    for index, (key, name, shape, color) in enumerate(view.instruments):
        x = 20 + spacing * (index + 0.5)
        level = view.levels[key]
        radius = min(48, spacing * 0.28) * (1 + level * 0.12 * math.sin(now * 5))
        fill = view.blend("#28354a", color, level)
        if level > 0.02:
            ring = radius + 13 + 4 * math.sin(now * 4)
            c.create_oval(x-ring, y-ring, x+ring, y+ring, outline=view.blend("#101827", color, level * 0.5), width=2)
        if use_graph_shapes and key in view.graph_shapes:
            view.draw_graph_icon(key, x, y, radius, fill)
        elif shape == "circle":
            c.create_oval(x-radius, y-radius, x+radius, y+radius, fill=fill, outline="")
        else:
            c.create_polygon(view.polygon_points(shape, x, y, radius), fill=fill, outline="")
        c.create_text(x, y+86, text=name, fill="#e4edf9", font=("", 13, "bold"))
        score = view.scores.get(key)
        text = f"{score:.1f} / 100" if score is not None and fresh else "—"
        detected = fresh and score is not None and score >= view.args.threshold
        c.create_text(x, y+115, text=text, fill=color if detected else "#9aabc2", font=("", 12))


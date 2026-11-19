"""映像パターン3。graphの楽器対応を使わず、元の図形を表示する。"""

from .common import draw_instruments


def draw(view, now, fresh):
    # この関数を編集すると、3キーで表示する映像を変更できます。
    # view.scores: 楽器ごとのスコア、now: 時刻、fresh: 結果が有効か。
    draw_instruments(view, now, fresh, use_graph_shapes=False)

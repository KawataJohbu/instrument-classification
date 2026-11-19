"""映像パターン2。内容が決まるまではscene1と同じ表示。"""

from .common import draw_instruments


def draw(view, now, fresh):
    # この関数を編集すると、2キーで表示する映像を変更できます。
    # view.scores: 楽器ごとのスコア、now: 時刻、fresh: 結果が有効か。
    draw_instruments(view, now, fresh)

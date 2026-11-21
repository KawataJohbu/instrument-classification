"""表示用モジュールだけをソースから読み直す。音声処理は対象外。"""

from pathlib import Path
import sys
from types import ModuleType


MODULES = (
    ("graph", "graph.py"),
    ("leaf_loop", "leaf_loop.py"),
    ("scenes", "scenes/__init__.py"),
    ("scenes.common", "scenes/common.py"),
    ("scenes.scene1", "scenes/scene1.py"),
    ("scenes.scene2", "scenes/scene2.py"),
    ("scenes.scene3", "scenes/scene3.py"),
    ("_orchestra_anime_live", "orchestra_anime.py"),
)


def restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def load_display_modules(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    # 先に全ファイルの構文を検証。pycの時刻判定に頼らず、保存内容を反映する。
    compiled = {name: compile((root/path).read_bytes(), str(root/path), "exec")
                for name, path in MODULES}
    previous = {name: sys.modules.get(name) for name, _ in MODULES}
    candidates = {}
    try:
        for name, path in MODULES:
            module = ModuleType(name)
            module.__file__ = str(root/path)
            module.__package__ = name.rpartition(".")[0]
            if name == "scenes":
                module.__package__ = "scenes"
                module.__path__ = [str(root/"scenes")]
            candidates[name] = module
            sys.modules[name] = module
        # 相対importが完了済みモジュールを参照できる順序で実行する。
        for name in ("graph", "leaf_loop", "scenes.common", "scenes.scene1",
                     "scenes.scene2", "scenes.scene3", "scenes", "_orchestra_anime_live"):
            exec(compiled[name], candidates[name].__dict__)
        for scene in candidates["scenes"].SCENES.values():
            if not callable(scene.draw):
                raise TypeError("Each scene must define draw(view, now, fresh)")
        return candidates, previous
    except Exception:
        restore_modules(previous)
        raise

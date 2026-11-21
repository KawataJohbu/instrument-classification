"""Scene 1: drifting clouds, falling leaves and bubbles on white."""
from dataclasses import dataclass, field
import math
import random

from .common import BACKGROUND

# Positions and travel distances are fractions of the window dimensions.
# Speeds are independent of frame rate. The reference aspect ratio is 1920x1920.
LEAVES = ("leaf_front", "leaf_diagonal", "leaf_side", "leaf_back")
CLOUDS = ("cloud1", "cloud3", "cloud2")
BUBBLES = (("bubbleS", 25), ("bubbleM", 40), ("bubbleL", 55), ("bubbleXL", 75))
INTERVALS = {"horn": 10.0, "cello": 1.5, "flute": 1.2}
PREVIEW_ALL = False  # 検知された楽器に対応する図形だけを出現させる。
LEAF_ROTATION_DEGREES = 90.0  # 落下中に画像中心から反時計回りに回る角度。
BUBBLE_TRAVEL_SECONDS = 20.0  # 長いほど、左下へゆっくり流れる。
NAMES = {"horn": "\u30db\u30eb\u30f3", "flute": "\u30d5\u30eb\u30fc\u30c8", "cello": "\u30c1\u30a7\u30ed"}


@dataclass
class Particle:
    instrument: str
    shape: str
    born: float
    y_start: float
    radius: float
    base_radius: float
    duration: float
    loop_phase: float = 0.
    sway_x: float = 0.
    sway_y: float = 0.
    sway_period_x: float = 4.
    sway_period_y: float = 5.
    sway_phase: float = 0.

    def position(self, now):
        progress = max(0., (now-self.born)/self.duration)
        if self.instrument == "horn":
            return -.10+1.20*progress, self.y_start
        if self.instrument == "cello":
            # In screen coordinates y grows downward: logarithmic fall,
            # initially steep and gradually bending toward the right.
            drop = .42*math.log1p(6*progress)/math.log(7)
            return -.06+1.18*progress, self.y_start+drop
        distance = 1.16*progress
        age = max(0., now-self.born)
        phase = self.sway_phase
        # 出現時の位置を保ちつつ、経過時間から連続した揺れを計算する。
        sway_x = self.sway_x*(math.sin(2*math.pi*age/self.sway_period_x+phase)-math.sin(phase))
        sway_y = self.sway_y*(math.cos(2*math.pi*age/self.sway_period_y+phase)-math.cos(phase))
        return (1.06-distance+sway_x,
                self.y_start+distance*math.tan(math.radians(40))+sway_y)


@dataclass
class SceneState:
    particles: list = field(default_factory=list)
    last_spawn: dict = field(default_factory=dict)
    active: set = field(default_factory=set)
    cloud_index: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def spawn(self, instrument, now):
        if instrument == "horn":
            shape = CLOUDS[self.cloud_index % 3]
            self.cloud_index += 1
            y_start = self.rng.uniform(.06, .28)
            # 現在の速度を基準に、下側だけ10%速くする。
            duration = 40.0 / 1.1 if y_start >= .17 else 40.0
            particle = Particle(instrument, shape, now, y_start, .1125, 130, duration)
        elif instrument == "cello":
            particle = Particle(instrument, "leaf_loop", now,
                                self.rng.uniform(.54, .72), .045, 115, 18.,
                                loop_phase=self.rng.random())
        else:
            shape, base_radius = self.rng.choice(BUBBLES)
            particle = Particle(instrument, shape, now, self.rng.uniform(.20, 2/3),
                                base_radius/1920, base_radius, BUBBLE_TRAVEL_SECONDS,
                                sway_x=self.rng.uniform(.012, .025),
                                sway_y=self.rng.uniform(.015, .030),
                                sway_period_x=self.rng.uniform(3., 6.),
                                sway_period_y=self.rng.uniform(3., 6.),
                                sway_phase=self.rng.uniform(0., 2*math.pi))
        self.particles.append(particle)
        self.last_spawn[instrument] = now

    def update(self, now, scores, threshold, fresh):
        active = (set(INTERVALS) if PREVIEW_ALL else
                  {key for key in INTERVALS if fresh and scores.get(key, 0) >= threshold})
        for key, interval in INTERVALS.items():
            if key in active and (key not in self.active or now-self.last_spawn.get(key, -math.inf) >= interval):
                self.spawn(key, now)
        self.active = active
        # Existing objects keep drifting after detection stops; new ones stop spawning.
        self.particles = [p for p in self.particles
                          if now-p.born <= p.duration and p.position(now)[1] < 1.15][-60:]


def draw(view, now, fresh):
    canvas = view.canvas
    view.graph_photos.clear()
    width, height = canvas.winfo_width(), canvas.winfo_height()
    if width < 2 or height < 2:
        return
    if not hasattr(view, "scene_states"):
        view.scene_states = {}
    if "scene1" not in view.scene_states:
        view.scene_states["scene1"] = SceneState()
    state = view.scene_states["scene1"]
    state.update(now, view.scores, view.args.threshold, fresh)
    active_keys = set()
    for particle in state.particles:
        x, y = particle.position(now)
        key = id(particle)
        active_keys.add(key)
        radius = particle.radius*min(width, height)
        if particle.instrument == "cello":
            age = max(0., now-particle.born)
            rotation = LEAF_ROTATION_DEGREES * min(1., age/particle.duration)
            view.draw_leaf_loop(x*width, y*height, radius,
                               age, particle.loop_phase, item_key=key, rotation=rotation)
        else:
            view.draw_graph_shape(particle.shape, x*width, y*height,
                                  radius, "black", particle.base_radius, item_key=key)
    view.finish_graph_items(active_keys)
    # Small black scores at bottom left; no other on-screen text.
    font_size = max(1, round(min(width, height)*.015))
    lines = []
    for key in ("horn", "flute", "cello"):
        score = view.scores.get(key)
        value = f"{score:.1f} / 100" if fresh and score is not None else "\u2014"
        lines.append(f"{NAMES[key]}  {value}")
    # White backing keeps scores legible if a leaf drifts behind them.
    snapshot = (width, height, font_size, "\n".join(lines))
    if view.scene1_overlay is None:
        backing = canvas.create_rectangle(0, 0, 0, 0, fill="white", outline="", tags="scene1-scores")
        label = canvas.create_text(0, 0, anchor="sw", fill="black", tags="scene1-scores")
        view.scene1_overlay = (backing, label, None)
    backing, label, previous = view.scene1_overlay
    if previous != snapshot:
        canvas.coords(backing, 0, height*.915, width*.22, height)
        canvas.coords(label, width*.015, height*.985)
        canvas.itemconfigure(label, text=snapshot[3], font=("", -font_size))
        view.scene1_overlay = (backing, label, snapshot)
    canvas.tag_raise("scene1-scores")

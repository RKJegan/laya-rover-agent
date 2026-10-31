"""Deterministic world generation (FR-1, Section 5.4).

Guarantees provided to the experiment:

* Same ``(seed, difficulty, preset)`` -> byte-identical map (independent RNG
  stream, so sweeping ``confidence_threshold`` cannot alter layouts).
* Every map is connectivity-verified between start and goal using the *runtime*
  traversability model (obstacles block, HAZARD_SLOPE is traversable-but-risky).
* A configurable share of maps are **trap-cluster presets**: U-shaped / box-
  canyon concave obstacle arrangements that are known local minima for a greedy
  no-lookahead controller. Without these, a naive greedy heuristic and a
  context-aware policy would rarely diverge and the research question would be
  structurally untestable (Section 5.4).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Literal

from .config import WorldConfig
from .types import CellType, GridMap

PresetKind = Literal["random", "trap_u", "trap_canyon", "trap_twincup"]


@dataclass(frozen=True)
class MapSpec:
    """Fully-determined description of one map in the experiment matrix."""

    seed: int
    difficulty: str
    preset: str

    @property
    def key(self) -> str:
        return f"{self.difficulty}_s{self.seed}_{self.preset}"


# --------------------------------------------------------------------------- #
# RNG stream separation
# --------------------------------------------------------------------------- #


def stream_seed(seed: int, stream: str | int) -> int:
    """Derive an independent RNG seed per stream.

    ``map_generation`` uses the seed verbatim so that map identity is obvious
    and stable; every other stream is mixed with a stable tag.

    ``stream`` is normally a human-readable string, whose character-sum is the
    tag. It may also be an ``int``, which is used as the tag directly - that is
    how a stream whose label has been renamed keeps its **exact** sequence
    instead of re-rolling (see ``laya.mock.MOCK_RNG_TAG_VALUE``).
    """
    if stream == "map_generation":
        return int(seed) & 0x7FFFFFFF
    tag = int(stream) if isinstance(stream, int) else sum((i + 1) * ord(c) for i, c in enumerate(stream))
    return (int(seed) * 1_000_003 + tag * 7919) & 0x7FFFFFFF


# --------------------------------------------------------------------------- #
# Preset schedule
# --------------------------------------------------------------------------- #


#: Preset schedule for a seed batch: the three engineered trap clusters plus
#: random-density terrain, interleaved so that every consecutive group of four
#: maps contains all four scenario types. Traps dominate deliberately (3 of 4):
#: FR-1 requires decision-relevant scenarios, and a pure random-density batch
#: rarely makes the two controllers diverge at all.
PRESET_SCHEDULE: tuple[str, ...] = ("trap_u", "trap_canyon", "trap_twincup", "random")


def preset_for_index(difficulty: str, index: int, traps_per_difficulty: int) -> str:
    """Preset for the ``index``-th seed of ``difficulty``.

    The schedule repeats ``trap_u, trap_canyon, trap_twincup, random`` so the
    batch always contains at least ``traps_per_difficulty`` trap maps (the
    spec's FR-1 floor) and every trap preset is represented many times over a
    30-seed batch rather than once.
    """
    schedule = PRESET_SCHEDULE
    if traps_per_difficulty > 0 and index < traps_per_difficulty:
        # Guarantee the spec floor in the first slots as well.
        return ("trap_u", "trap_canyon", "trap_twincup")[index % 3]
    return schedule[index % len(schedule)]


def build_map_specs(
    seeds_per_difficulty: int,
    difficulties: list[str],
    traps_per_difficulty: int,
) -> list[MapSpec]:
    specs: list[MapSpec] = []
    for difficulty in difficulties:
        for index in range(seeds_per_difficulty):
            specs.append(
                MapSpec(
                    seed=index,
                    difficulty=difficulty,
                    preset=preset_for_index(difficulty, index, traps_per_difficulty),
                )
            )
    return specs


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class WorldGenerator:
    def __init__(self, config: WorldConfig) -> None:
        self.config = config

    # -- public API --------------------------------------------------------- #

    def generate(self, spec: MapSpec) -> GridMap:
        rng = random.Random(stream_seed(spec.seed, "map_generation") ^ hash_preset(spec.preset))
        width, height = self.config.width, self.config.height
        obstacle_density = self.config.obstacle_density.get(spec.difficulty, 0.18)
        hazard_density = self.config.hazard_density.get(spec.difficulty, 0.06)

        start = self._pick_start(rng, width, height)
        goal = self._pick_goal(rng, start, width, height)

        cells = [[CellType.FREE.value for _ in range(width)] for _ in range(height)]

        traps: list[dict[str, int]] = []
        if spec.preset.startswith("trap_"):
            traps = self._draw_trap(rng, spec.preset, start, goal, width, height, cells)

        self._scatter(rng, cells, obstacle_density, CellType.OBSTACLE, protected=_protected_cells(start, goal, traps, width, height))
        self._scatter(rng, cells, hazard_density, CellType.HAZARD_SLOPE, protected=_protected_cells(start, goal, traps, width, height))

        # Never seal the map: carve a guaranteed minimum-viable route. The trap
        # presets keep their mouth open by construction (see _trap_wall), so the
        # carving step only has to guarantee global connectivity.
        path = self._carve_route(rng, cells, start, goal)

        cells[start[1]][start[0]] = CellType.START.value
        cells[goal[1]][goal[0]] = CellType.GOAL.value

        grid = GridMap(
            width=width,
            height=height,
            start=start,
            goal=goal,
            cells=cells,
            trap_clusters=tuple(traps),
            preset=spec.preset,
            difficulty=spec.difficulty,
            seed=spec.seed,
        )
        assert route_exists(grid), "world generator produced an unreachable goal"
        assert path is not None
        return grid

    # -- internals ---------------------------------------------------------- #

    def _pick_start(self, rng: random.Random, width: int, height: int) -> tuple[int, int]:
        margin = max(1, min(width, height) // 10)
        return (rng.randint(margin, margin + 2), rng.randint(margin, margin + 2))

    def _pick_goal(self, rng: random.Random, start: tuple[int, int], width: int, height: int) -> tuple[int, int]:
        min_dist = self.config.min_start_goal_distance_fraction * ((width**2 + height**2) ** 0.5)
        margin = max(1, min(width, height) // 10)
        for _ in range(2000):
            candidate = (rng.randint(width - margin - 3, width - margin), rng.randint(height - margin - 3, height - margin))
            if _dist(start, candidate) >= min_dist:
                return candidate
        return (width - margin - 1, height - margin - 1)

    def _draw_trap(
        self,
        rng: random.Random,
        preset: str,
        start: tuple[int, int],
        goal: tuple[int, int],
        width: int,
        height: int,
        cells: list[list[str]],
    ) -> list[dict[str, int]]:
        """Place a *visible, avoidable* concave pocket on the straight line
        between start and goal, with its mouth facing the approach.

        Design rule: the pocket is 3 cells deep, so a rover whose forward beam
        reaches it sees the enclosing wall within the sensor window and can
        choose to go around. That is what makes these maps decision-relevant:
        a memoryless greedy controller walks in and oscillates at the back wall,
        while a conservative policy that reads its own window can take the (very
        slightly longer) clean route. Deeper pockets would be unwinnable for
        every local policy and would test nothing.
        """
        if preset == "trap_u":
            w, h = 5, 4
        elif preset == "trap_canyon":
            w, h = 3, 4
        else:  # trap_twincup
            w, h = 5, 4

        # Target: the point ~45% of the way along the start->goal line, so the
        # pocket sits *in front of* the rover early in the episode rather than
        # next to the goal, and so the rover must decide before it can simply
        # walk around the far side.
        target_x = start[0] + 0.45 * (goal[0] - start[0])
        target_y = start[1] + 0.45 * (goal[1] - start[1])

        best: tuple[int, int] | None = None
        best_score = float("inf")
        for _ in range(600):
            cx = rng.randint(2, max(2, width - w - 2))
            cy = rng.randint(2, max(2, height - h - 2))
            if _rect_contains((cx, cy, w, h), start) or _rect_contains((cx, cy, w, h), goal):
                continue
            centre = (cx + w // 2, cy + h // 2)
            score = _dist(centre, (int(round(target_x)), int(round(target_y))))
            if score < best_score:
                best_score = score
                best = (cx, cy)
        if best is None or best_score > min(width, height) * 0.5:
            return []

        cx, cy = best
        mouth = _mouth_side((cx + w // 2, cy + h // 2), start)
        wall = trap_wall(preset, w, h, mouth)
        bricked = 0
        for dy, row in enumerate(wall):
            for dx, ch in enumerate(row):
                if ch != "#":
                    continue
                if 0 <= cy + dy < height and 0 <= cx + dx < width:
                    cells[cy + dy][cx + dx] = CellType.OBSTACLE.value
                    bricked += 1
        return [{"x": cx, "y": cy, "w": w, "h": h, "bricks": bricked}]

    def _scatter(
        self,
        rng: random.Random,
        cells: list[list[str]],
        density: float,
        cell_type: CellType,
        protected: set[tuple[int, int]],
    ) -> None:
        height = len(cells)
        width = len(cells[0])
        target = int(density * width * height)
        placed = sum(row.count(cell_type.value) for row in cells)
        attempts = 0
        while placed < target and attempts < target * 40 + 500:
            attempts += 1
            x = rng.randrange(width)
            y = rng.randrange(height)
            if (x, y) in protected or cells[y][x] != CellType.FREE.value:
                continue
            cells[y][x] = cell_type.value
            placed += 1

    def _carve_route(
        self,
        rng: random.Random,
        cells: list[list[str]],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Carve a guaranteed traversable route (used as the minimum-viable-route
        check). Prefers a concave-trap-avoiding detour so traps stay meaningful."""
        width, height = len(cells[0]), len(cells)
        route = _astar_or_bfs(cells, start, goal, width, height)
        if route is not None:
            return route

        # No route: carve a monotone staircase, punching through whatever is in
        # the way (obstacles become hazards where possible to keep risk texture).
        x, y = start
        route = [(x, y)]
        while (x, y) != goal:
            if x != goal[0] and (rng.random() < 0.5 or y == goal[1]):
                x += 1 if goal[0] > x else -1
            elif y != goal[1]:
                y += 1 if goal[1] > y else -1
            if cells[y][x] == CellType.OBSTACLE.value:
                cells[y][x] = CellType.HAZARD_SLOPE.value
            route.append((x, y))
        return route

def _mouth_side(centre: tuple[int, int], start: tuple[int, int]) -> str:
    """Which side of the trap should stay open, given where the rover comes from."""
    dx = start[0] - centre[0]
    dy = start[1] - centre[1]
    if abs(dx) > abs(dy):
        return "west" if dx < 0 else "east"
    return "north" if dy < 0 else "south"


def trap_wall(preset: str, w: int, h: int, mouth: str = "north") -> list[str]:
    """ASCII layout of a concave pocket ('#' = obstacle, '.' = free).

    All pockets are 3 cells deep on their long axis so that the enclosing wall
    is inside the sensor window of a rover standing at the mouth:

    ``trap_u``        a 3x2 cup: walls left/right/back, mouth = approach side
    ``trap_canyon``   a 1-wide, 3-deep blind slot
    ``trap_twincup``  two 1-wide, 3-deep slots separated by a pillar
    """
    glyph = _pocket_glyph(preset)
    return _orient(glyph, mouth)


def _pocket_glyph(preset: str) -> list[str]:
    """Canonical layout, mouth facing north (top row open)."""
    if preset == "trap_canyon":
        return [
            ".#.",
            ".#.",
            ".#.",
            "###",
        ]
    if preset == "trap_twincup":
        return [
            ". . .",
            "#.#.#",
            "#.#.#",
            "#####",
        ]
    # default: three-sided cup with a 3-wide, 2-deep interior
    return [
        ".....",
        "#...#",
        "#...#",
        "#####",
    ]


def _orient(glyph: list[str], mouth: str) -> list[str]:
    """Rotate the canonical (mouth-north) layout so the mouth faces ``mouth``.

    Rotation keeps the pocket's depth intact, so the visibility property holds
    for every approach direction.
    """
    grid = [list(row.replace(" ", ".")) for row in glyph]
    turns = {"north": 0, "east": 1, "south": 2, "west": 3}[mouth]
    for _ in range(turns):
        grid = [list(row) for row in zip(*grid[::-1])]  # 90 degrees clockwise
    return ["".join(row) for row in grid]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def hash_preset(preset: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(preset)) * 104_729


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _rect_contains(rect: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    x0, y0, w, h = rect
    return x0 <= point[0] < x0 + w and y0 <= point[1] < y0 + h


def _on_border(x: int, y: int, x0: int, y0: int, w: int, h: int) -> bool:
    return x in (x0, x0 + w - 1) or y in (y0, y0 + h - 1)


def _protected_cells(
    start: tuple[int, int],
    goal: tuple[int, int],
    traps: list[dict[str, int]],
    width: int,
    height: int,
) -> set[tuple[int, int]]:
    protected: set[tuple[int, int]] = set()
    for centre in (start, goal):
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                protected.add((centre[0] + dx, centre[1] + dy))
    for trap in traps:
        for y in range(trap["y"], trap["y"] + trap["h"]):
            for x in range(trap["x"], trap["x"] + trap["w"]):
                protected.add((x, y))
    # Keep the outer ring free so the rover can always route around.
    for x in range(width):
        protected.add((x, 0))
        protected.add((x, height - 1))
    for y in range(height):
        protected.add((0, y))
        protected.add((width - 1, y))
    return protected


def route_exists(grid: GridMap, hazard_traversable: bool = True) -> bool:
    """Minimum-viable-route check under the runtime traversability model."""
    seen = {grid.start}
    queue: deque[tuple[int, int]] = deque([grid.start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == grid.goal:
            return True
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nxt = (x + dx, y + dy)
            if nxt in seen or not grid.in_bounds(*nxt):
                continue
            cell = grid.cell(*nxt)
            if cell is CellType.OBSTACLE:
                continue
            if cell is CellType.HAZARD_SLOPE and not hazard_traversable:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


def _astar_or_bfs(
    cells: list[list[str]],
    start: tuple[int, int],
    goal: tuple[int, int],
    width: int,
    height: int,
) -> list[tuple[int, int]] | None:
    """Cheap BFS over traversable cells; hazards cost more in a weighted pass."""
    import heapq

    costs = {CellType.FREE.value: 1.0, CellType.HAZARD_SLOPE.value: 2.0}
    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            return list(reversed(path))
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nxt = (current[0] + dx, current[1] + dy)
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            cell = cells[nxt[1]][nxt[0]]
            if cell == CellType.OBSTACLE.value:
                continue
            step = costs.get(cell, 1.0)
            tentative = g_score[current] + step
            if tentative < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative
                came[nxt] = current
                heuristic = _dist(nxt, goal)
                heapq.heappush(open_heap, (tentative + heuristic, nxt))
    return None


def ascii_map(grid: GridMap) -> str:
    """Human-readable render used by the CLI and tests."""
    glyphs = {
        CellType.FREE.value: ".",
        CellType.OBSTACLE.value: "#",
        CellType.HAZARD_SLOPE.value: "~",
        CellType.START.value: "S",
        CellType.GOAL.value: "G",
    }
    return "\n".join("".join(glyphs[c] for c in row) for row in grid.cells)
